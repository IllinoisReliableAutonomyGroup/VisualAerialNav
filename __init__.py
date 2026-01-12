"""Quadrotor simulation package with plant, controller, and visualization utilities."""

from .plant import QuadrotorParams, QuadrotorState, ControlInput, QuadrotorPlant
from .controller import (
    TrajectoryCommand,
    Controller,
    GeometricController,
    SE3PositionController,
    RandomController,
    PBVSController,
    PBVSSmoothedController,
)
from .simulator import SimulationLog, Simulator
from .scene import CameraObservation
# Note: StateEstimator, IdentityEstimator, NoisyStateEstimator, CameraLogger
# are not currently implemented in estimator.py

__all__ = [
    "QuadrotorParams",
    "QuadrotorState",
    "ControlInput",
    "QuadrotorPlant",
    "TrajectoryCommand",
    "Controller",
    "GeometricController",
    "SE3PositionController",
    "RandomController",
    "PBVSController",
    "PBVSSmoothedController",
    "CameraObservation",
    "SimulationLog",
    "Simulator",
]
