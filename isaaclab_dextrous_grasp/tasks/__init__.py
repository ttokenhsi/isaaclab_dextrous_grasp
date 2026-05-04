"""Task registry. Importing the submodules has the side effect of calling
``gymnasium.register`` so that IsaacLab can find the tasks by their gym ID."""

from . import allegro_relocate  # noqa: F401
