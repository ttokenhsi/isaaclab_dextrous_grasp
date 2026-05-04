"""Standalone IsaacLab port of the ViViDex (SAPIEN) UR5 + Allegro YCB grasp task.

Importing this package registers the gym task IDs defined in
``isaaclab_dextrous_grasp.tasks.allegro_relocate``.
"""

from . import tasks  # noqa: F401  (registers gym IDs as a side effect)

__version__ = "0.1.0"
