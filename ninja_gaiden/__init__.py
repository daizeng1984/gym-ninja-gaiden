"""Registration code of Gym environments in this package."""
from ninja_gaiden.ninja_env import NinjaGaiden
from ninja_gaiden.ninja_proc import NesGymProc


# define the outward facing API of this package
__all__ = [
    NinjaGaiden.__name__,
    NesGymProc.__name__,
]
