"""Gate perception from low-resolution agent cameras: a CNN that maps each
agent's camera image to whether a gate is in view and the goal bearing, on
the emergent_leaders motion model."""
from .camera import CameraParams, render
from .network import GoalNet
from .simulation import VisionSimulation

__all__ = ["CameraParams", "GoalNet", "VisionSimulation", "render"]
