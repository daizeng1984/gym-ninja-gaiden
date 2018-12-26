"""
An OpenAI Gym interface to the NES game <TODO: Game Name>
from https://github.com/Kautenja/nes-py/wiki/Creating-Environments
"""
import os
import gym
from nes_py import NESEnv
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


# Register the envs
gym.envs.registration.register(
    id='ninja-gaiden-v0',
    entry_point='ninja_gaiden:NinjaGaiden',
    max_episode_steps=9999999,
    reward_threshold=32000,
    nondeterministic=True,
)

# actions for more complex movement
DEFAULT_MOVEMENT = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up']
]


def _make_ninja_gaiden_gym(action_space=DEFAULT_MOVEMENT):
    env = gym.make('ninja-gaiden-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, action_space)
    return env


class NinjaGaiden(NESEnv):
    """An OpenAI Gym interface to the NES game <TODO: Game Name>"""

    def __init__(self):
        """Initialize a new <TODO: Game Name> environment."""
        super(NinjaGaiden, self).__init__(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ninja-gaiden-u.nes'),
            frames_per_step=4,
            max_episode_steps=99999
        )
        # setup any variables to use in the below callbacks here
        self._time_last = 0
        # setup a variable to keep track of the last frames x position
        self._x_position_last = 0
        self._hp_last = 0
        self._lives_last = 0
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # stall for a frame
        self.step(0)
        # create a backup state to restore from on subsequent calls to reset
        self._backup()
        

    def _is_game_over(self):
        return self._time >= 255 or self._time <= 0 or (self._lives_last - self._lives) > 0
    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button, start = 8 nop = 0
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts (timer start)
        # TODO: Hackily hack
        while self._time >= 255 or self._time <= 0:
            # press and release the start button
            self._frame_advance(8)
            self._frame_advance(0)

    

    def _will_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to perform setup before and episode resets.
        # the method returns None
        pass

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to access the RAM of the emulator 
        # and perform setup for each episode. 
        # the method returns None
        self._screen_x_last = self._screen_x
        self._lives_last = self._lives
        self._time_last = self._time
        self._hp_last = self._hp
        pass

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        pass

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # Update all states
        moved = max(0, self._screen_x - self._screen_x_last) * 4
        self._screen_x_last = max(self._screen_x_last, self._screen_x)

        # time
        time = self._time - self._time_last
        self._time_last = self._time

        # hp
        hpdiff = self._hp - self._hp_last
        self._hp_last = self._hp
        
        ret= min(15, max(-15, moved + time + hpdiff))
        # print('reward: {} moved: {} time: {} hp: {} screen: {} '.format(ret, moved, time, hpdiff, self._screen_x))
        
        return ret

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self._is_game_over()

    @property
    def _time(self):
        return self._read_mem(0x0063)
    @property
    def _ninja_pts(self):
        return self._read_mem(0x0064)
    @property
    def _score(self):
        return self._read_mem(0x0061)*256 + self._read_mem(0x0060)
    @property
    def _hp(self):
        return self._read_mem(0x0065)
    @property
    def _boss_hp(self):
        return self._read_mem(0x0066)
    @property
    def _level(self):
        return self._read_mem(0x006D)
    @property
    def _stage(self):
        return self._read_mem(0x006E)
    @property
    def _lives(self):
        return self._read_mem(0x0076)
    @property
    def _player_y_delta(self):
        return self._read_mem(0x0089) + self._read_mem(0x0087)/256
    @property
    def _player_y_pos(self):
        return self._read_mem(0x008A)
    def to8bitSigned(self, num): 
        mask7 = 128 #Check 8th bit ~ 2^8
        mask2s = 127 # Keep first 7 bits
        if (mask7 & num == 128): #Check Sign (8th bit)
            num = -((~int(num) + 1) & mask2s) #2's complement
        return num

    @property
    def _screen_x(self):
        # block pos self._read_mem(0x0053) * 256 * 256 + 
        return self.to8bitSigned(self._read_mem(0x0052)) *256 + self._read_mem(0x0051) + \
            (self._read_mem(0x0050)/256)
    @property
    def _player_state(self):
        return self._read_mem(0x0084)
    @property
    def _player_x_delta(self):
        return self._read_mem(0x00AD) + self._read_mem(0x00AC)/256
    @property
    def _player_x_pos(self):
        return self._read_mem(0x0086) + self._read_mem(0x0085)/256
    @property
    def _invincibility_timeout(self):
        # invulnerability timer, runs 60-0 starting when Ryu takes damage
        return self._read_mem(0x0095)
    @property
    def _weapon(self):
	# 00 - None
	# 80 - Fire Wheel
	# 81 - Throwing Star
	# 82 - Windmill Star
	# 83 - Hourglass (Unusable)
	# 84 - Fire Ring
	# 85 - Jump-N-Slash
        return self._read_mem(0x00C9)
    @property
    def _seconds_timeout(self):
        return self._read_mem(0x04C7)*256 + self._read_mem(0x04C8)
    
    def _read_mem_ext(self, start):
        ret = ""
        for i in range(0, 16):
            ret = ret + "," + str((self._read_mem(i + start)))
        return ret
    def _get_info(self):
        """Return the info after a step occurs."""
        return {
            'time': self._time,
            'level': self._level,
            'stage': self._stage,
            'lives' : self._lives,
            'x_pos' : self._player_x_pos,
            'x_speed' : self._player_x_delta,
            'y_pos' : self._player_y_pos,
            'y_speed' : self._player_y_delta,
            'screen_x' : self._screen_x,
            'hp' : self._hp,
            'state': self._player_state
        }


# explicitly define the outward facing API for the module
__all__ = [NinjaGaiden.__name__]
