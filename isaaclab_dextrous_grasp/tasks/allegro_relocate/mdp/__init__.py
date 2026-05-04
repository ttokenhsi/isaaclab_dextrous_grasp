"""MDP terms for the AllegroUR5 YCB relocate task.

The submodules expose:

* :mod:`actions` -- :class:`IKHandActionCfg` / :class:`IKHandAction` (22-d).
* :mod:`observations` -- functions used by the 4 ObsTerms.
* :mod:`rewards` -- functions used by the 7 RewTerms.
* :mod:`terminations` -- functions used by the 4 DoneTerms.
* :mod:`events` -- :func:`reset_trajectory_state`.
"""

from .actions import IKHandAction, IKHandActionCfg

from . import events  # noqa: F401
from . import observations  # noqa: F401
from . import rewards  # noqa: F401
from . import terminations  # noqa: F401
