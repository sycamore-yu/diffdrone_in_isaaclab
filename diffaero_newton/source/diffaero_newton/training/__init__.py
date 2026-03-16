"""Training layer for SHAC-style reinforcement learning."""

from diffaero_newton.training.shac import SHAC, SHACAgent
from diffaero_newton.training.mashac import MASHAC, MASHACAgent
from diffaero_newton.training.buffer import RolloutBuffer, StateRolloutBuffer
