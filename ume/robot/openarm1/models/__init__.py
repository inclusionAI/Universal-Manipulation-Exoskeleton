"""Robot models and MuJoCo descriptions for OpenArm simulation.

This module provides paths and constants for accessing robot model files,
particularly the MuJoCo XML model used for simulation.
"""

from pathlib import Path

OPENARM_MODEL_PATH = Path(__file__).parent / "openarm.xml"

__all__ = ["OPENARM_MODEL_PATH"]
