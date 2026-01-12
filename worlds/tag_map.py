"""Utility to extract marker poses and colors from a URDF into a TagMapDictionary."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
from typing import List, Tuple

import numpy as np

try:
    from .math_utils import rpy_to_rot, rot_to_rpy
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from math_utils import rpy_to_rot, rot_to_rpy


def _parse_xyz_rpy(elem: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])
    if "xyz" in elem.attrib:
        xyz = np.array([float(v) for v in elem.attrib["xyz"].split()])
    if "rpy" in elem.attrib:
        rpy = np.array([float(v) for v in elem.attrib["rpy"].split()])
    return xyz, rpy


def _compose(xyz1: np.ndarray, rpy1: np.ndarray, xyz2: np.ndarray, rpy2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two poses given as (xyz, rpy). This is used to compute the world pose of a tag given its visual and joint origins."""
    R1 = rpy_to_rot(rpy1)
    R2 = rpy_to_rot(rpy2)
    R = R1 @ R2
    t = xyz1 + R1 @ xyz2
    return t, rot_to_rpy(R)


def load_tag_map(urdf_path: str) -> List[dict]:
    """Construct a tag map dictionary from the given URDF file which contains for each tag its id, name, position, rpy, and color (if specified)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {}
    for joint in root.findall("joint"):
        child_elem = joint.find("child")
        if child_elem is None or "link" not in child_elem.attrib:
            continue
        child_name = child_elem.attrib["link"]
        joints[child_name] = joint

    tag_map = []
    uid = 0
    for name, link in links.items():
        if name == "world":
            continue
        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None or (geom.find("box") is None and geom.find("mesh") is None):
            # Accept either box or mesh geometry for a tag visual
            continue

        vis_origin_elem = visual.find("origin")
        vis_xyz = np.zeros(3)
        vis_rpy = np.zeros(3)
        if vis_origin_elem is not None:
            vis_xyz, vis_rpy = _parse_xyz_rpy(vis_origin_elem)

        joint = joints.get(name)
        if joint is None:
            continue
        joint_origin_elem = joint.find("origin")
        joint_xyz = np.zeros(3)
        joint_rpy = np.zeros(3)
        if joint_origin_elem is not None:
            joint_xyz, joint_rpy = _parse_xyz_rpy(joint_origin_elem)

        world_xyz, world_rpy = _compose(joint_xyz, joint_rpy, vis_xyz, vis_rpy)

        color = None
        mat = visual.find("material")
        if mat is not None:
            color_elem = mat.find("color")
            if color_elem is not None and "rgba" in color_elem.attrib:
                color = [float(v) for v in color_elem.attrib["rgba"].split()]

        tag_map.append(
            {
                "id": uid,
                "name": name,
                "position": world_xyz.tolist(),
                "rpy": world_rpy.tolist(),
                "color": color,
            }
        )
        uid += 1
    return tag_map


def _apply_tag_textures(body_id: int, textures_dir: str) -> None:
    """Apply textures to links whose names end with a tag id."""
    import pybullet as p

    if not os.path.isdir(textures_dir):
        return

    def _apply(link_idx: int, link_name: str) -> None:
        suffix = "".join(ch for ch in link_name if ch.isdigit())
        if not suffix:
            return
        tex_path = os.path.join(textures_dir, f"tag_{suffix}.png")
        if not os.path.exists(tex_path):
            return
        tex_id = p.loadTexture(tex_path)
        p.changeVisualShape(
            body_id,
            link_idx,
            textureUniqueId=tex_id,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0],
        )

    base_name = p.getBodyInfo(body_id)[0].decode("utf-8")
    _apply(-1, base_name)
    for link_idx in range(p.getNumJoints(body_id)):
        link_name = p.getJointInfo(body_id, link_idx)[12].decode("utf-8")
        _apply(link_idx, link_name)


def visualize_tag_map(urdf_path: str, tags: List[dict]) -> None:
    """Render the tag map, forcing AprilTag textures to cover each marker."""

    urdf_path = os.path.abspath(urdf_path)
    base_dir = os.path.dirname(__file__)
    tag_size = 0.6  # meters
    quad_obj = os.path.join(base_dir, "tag_quad.obj")
    textures_dir = os.path.abspath(os.path.join(base_dir, "..", "textures", "tags_36h11"))

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(base_dir)
    p.setAdditionalSearchPath(textures_dir)

    if not os.path.exists(urdf_path):
        print(f"URDF not found: {urdf_path}")
        return

    env_id = p.loadURDF(urdf_path)
    _apply_tag_textures(env_id, textures_dir)

    for tag in tags:
        pos = np.asarray(tag["position"])
        rpy = np.asarray(tag["rpy"])
        label = (
            f"id:{tag['id']} pos:({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
            f"rpy:({rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f})"
        )
        print(label)
        p.addUserDebugText(label, pos.tolist(), textColorRGB=[0, 0, 0], textSize=0.8, lifeTime=0)

        tex_path = os.path.join(textures_dir, f"tag_{tag['id']}.png")
        if not (os.path.exists(tex_path) and os.path.exists(quad_obj)):
            continue

        tex_id = p.loadTexture(tex_path)
        quat = p.getQuaternionFromEuler(rpy.tolist())
        normal = rpy_to_rot(rpy)[:, 2]
        offset_pos = (pos + 0.002 * normal).tolist()
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=quad_obj,
            meshScale=[tag_size, tag_size, tag_size],
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0],
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=offset_pos,
            baseOrientation=quat,
        )
        p.changeVisualShape(body, -1, textureUniqueId=tex_id)

    while p.isConnected():
        p.stepSimulation()


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    urdf = os.path.join(here, "marker_field_apriltags.urdf")
    TagMapDictionary = load_tag_map(urdf)
    print("Loaded", len(TagMapDictionary), "tags")
    for tag in TagMapDictionary:
        pos = [round(v, 2) for v in tag["position"]]
        rpy = [round(v, 2) for v in tag["rpy"]]
        print({**tag, "position": pos, "rpy": rpy})
    visualize_tag_map(urdf, TagMapDictionary)
