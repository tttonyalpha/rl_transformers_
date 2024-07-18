# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import bsuite
# from bsuite.utils import gym_wrapper

import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

import bsuite
# from bsuite.utils import gym_wrapper
from gym import spaces
import gym
import dm_env
from dm_env import specs
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import bsuite
from bsuite.utils import gym_wrapper

# OpenAI gym step format = obs, reward, is_finished, other_info
# _GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


# class GymFromDMEnv(gym.Env):
#     """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""
#     metadata = {'render.modes': ['human', 'rgb_array']}
#     def __init__(self, env: dm_env.Environment):
#         self._env = env  # type: dm_env.Environment
#         self._last_observation = None  # type: Optional[np.ndarray]
#         self.viewer = None
#         self.game_over = False  # Needed for Dopamine agents.

#         obs_spec = self._env.observation_spec()  # type: specs.Array
#         if isinstance(obs_spec, specs.BoundedArray):
#             self.observation_space = spaces.Box(
#                 low=float(obs_spec.minimum),
#                 high=float(obs_spec.maximum),
#                 shape=obs_spec.shape,
#                 dtype=obs_spec.dtype)
#         self.observation_space = spaces.Box(
#             low=-float('inf'),
#             high=float('inf'),
#             shape=obs_spec.shape,
#             dtype=obs_spec.dtype)

#     def step(self, action: int) ->  Tuple[np.ndarray, float, bool, Dict[str, Any]]:
#         timestep = self._env.step(action)
#         self._last_observation = timestep.observation
#         reward = timestep.reward or 0.
#         if timestep.last():
#           self.game_over = True
#         return timestep.observation, reward, timestep.last(), {}

#     def reset(self) -> np.ndarray:
#         self.game_over = False
#         timestep = self._env.reset()
#         self._last_observation = timestep.observation
#         return timestep.observation

#     def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
#         if self._last_observation is None:
#           raise ValueError('Environment not ready to render. Call reset() first.')

#         if mode == 'rgb_array':
#           return self._last_observation

#         if mode == 'human':
#           if self.viewer is None:
#             # pylint: disable=import-outside-toplevel
#             # pylint: disable=g-import-not-at-top
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.SimpleImageViewer()
#           self.viewer.imshow(self._last_observation)
#           return self.viewer.isopen


#     @property
#     def action_space(self) -> spaces.Discrete:
#         action_spec = self._env.action_spec()  # type: specs.DiscreteArray
#         return spaces.Discrete(action_spec.num_values)

#     @property
#     def reward_range(self) -> Tuple[float, float]:
#         reward_spec = self._env.reward_spec()
#         if isinstance(reward_spec, specs.BoundedArray):
#           return reward_spec.minimum, reward_spec.maximum
#         return -float('inf'), float('inf')

# class BsuiteWrapper_old():
#     """
#     This class wraps bsuite enviroments
#     Available Environments:

#     * memory_len/0
#     * discounting_chain/0
#     """
#     def __init__(self, env_name, reset_params = None, realtime_mode = False) -> None:
#         """Instantiates the bsuite environment.
        
#         Arguments:
#             env_name {string} -- Name of the bsuite environment
#             reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
#             realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
#         Attributes: 
#             observation_space_shape {tuple} -- Shape of the observation space
            
#         """

#         self._rewards = []
#         self.t = 0

#         if reset_params is None:
#             self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
#         else:
#             self._default_reset_params = reset_params

#         # TO-DO: add another envs
#         # if you want to use another popgym enviroment just add desired rows below

#         self._env = GymFromDMEnv(bsuite.load_from_id(env_name))
#         self._env.seed(self._default_reset_params['start-seed'])

#         if isinstance(self._env.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
#             self._env.observation_space.obs_shape = self._env.observation_space.shape 
#             print(self._env.observation_space.obs_shape)
#             if len(self._env.observation_space.obs_shape) > 1:
#                 self._env.observation_space.obs_type = 'image'
#             else:
#                 self._env.observation_space.obs_type = 'vector'
#         elif isinstance(self._env.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
#             # discrete observation
#             self._env.observation_space.obs_shape = (1, )
#             self._env.observation_space.obs_type = 'discrete'
#         elif isinstance(self._env.observation_space, Iterable) and isinstance(self._env.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
#             # multi discrete observation - like discrete vector 
#             self._env.observation_space.obs_shape = (1 for obs_space in env.observation_space )
#             self._env.observation_space.obs_type = 'iterable_discrete'
#         elif isinstance(self._env.observation_space, Iterable) and isinstance(self._env.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
#             # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
#             raise NotImplementedError

#     @property
#     def observation_space(self):
#         """Returns the shape of the observation space of the agent."""
#         return self._env.observation_space

#     @property
#     def action_space(self):
#         """Returns the shape of the action space of the agent."""
#         return self._env.action_space

#     @property
#     def max_episode_steps(self):
#         """Returns the maximum number of steps that an episode can last."""
#         self._env.reset()

#         try:
#             max_lenght = self._env._env.bsuite_num_episodes
#         except Exception as e:
#             print(f'ERROR: {e}')
#             #TO-DO: define for envs when None!!!
#             max_lenght = 96 # define 
#         return max_lenght

#     def reset(self, reset_params = None):
#         """Resets the environment.
        
#         Keyword Arguments:
#             reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
#         Returns:
#             {numpy.ndarray} -- Visual observation
#         """

#         self.t = 0
#         self._rewards = []

#         # Reset the environment to retrieve the initial observation
#         obs = self._env.reset()
#         self._env.seed(self._default_reset_params['start-seed'])


#         if isinstance(obs, int):
#             obs = np.array([obs, ])
#         elif isinstance(obs, Iterable):
#             obs = np.array(obs)

#         return obs

#     def step(self, action):
#         """Runs one timestep of the environment's dynamics.
        
#         Arguments:
#             action {list} -- The to be executed action
        
#         Returns:
#             {numpy.ndarray} -- Visual observation
#             {float} -- (Total) Scalar reward signaled by the environment
#             {bool} -- Whether the episode of the environment terminated
#             {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
#         """
#         if isinstance(action, list):
#             if len(action) == 1:
#                 action = action[0]
#         obs, reward, terminated, info  = self._env.step(action)
#         self._rewards.append(reward)


#         if isinstance(obs, int):
#             obs = np.array([obs, ])
#         elif isinstance(obs, Iterable):
#             obs = np.array(obs)

#         if self.t == self.max_episode_steps - 1:
#             terminated = True


#         if terminated:
#             info = {"reward": sum(self._rewards),
#                     "length": len(self._rewards)}

#         self.t += 1

#         return obs, reward, terminated, info
    
#     def render(self):
#         """Renders the environment."""
#         self._env.render()

#     def close(self):
#         """Shuts down the environment."""
#         self._env.close()


# """Simple diagnostic discounting challenge.

# Observation is two pixels: (context, time_to_live)

# Context will only be -1 in the first step, then equal to the action selected in
# the first step. For all future decisions the agent is in a "chain" for that
# action. Reward of +1 come  at one of: 1, 3, 10, 30, 100

# However, depending on the seed, one of these chains has a 10% bonus.
# """

# from typing import Any, Dict, Optional

# from bsuite.environments import base
# from bsuite.experiments.discounting_chain import sweep

# import dm_env
# from dm_env import specs
# import numpy as np


# class DiscountingChain(base.Environment):
#     """Discounting Chain environment."""

#     def __init__(self, mapping_seed: Optional[int] = None):
#         """Builds the Discounting Chain environment.

#         Args:
#           mapping_seed: Optional integer, specifies which reward is bonus.
#         """
#         super().__init__()
#         self._episode_len = 100
#         self._reward_timestep = [1, 3, 10, 30, 100]
#         self._n_actions = len(self._reward_timestep)
#         if mapping_seed is None:
#             mapping_seed = np.random.randint(0, self._n_actions)
#         else:
#             mapping_seed = mapping_seed % self._n_actions

#         self._rewards = np.ones(self._n_actions)
#         self._rewards[mapping_seed] += 0.1

#         self._timestep = 0
#         self._context = -1

#         self.bsuite_num_episodes = 10000 #sweep.NUM_EPISODES

#     def _get_observation(self):
#         obs = np.zeros(shape=(1, 2), dtype=np.float32)
#         obs[0, 0] = self._context
#         obs[0, 1] = self._timestep / self._episode_len
#         return obs

#     def _reset(self) -> dm_env.TimeStep:
#         self._timestep = 0
#         self._context = -1
#         observation = self._get_observation()
#         return dm_env.restart(observation)

#     def _step(self, action: int) -> dm_env.TimeStep:
#         if self._timestep == 0:
#             self._context = action

#         self._timestep += 1
#         if self._timestep == self._reward_timestep[self._context]:
#             reward = self._rewards[self._context]
#         else:
#             reward = 0.0

#         observation = self._get_observation()
#         if self._timestep == self._episode_len:
#             return dm_env.termination(reward=reward, observation=observation)
#         return dm_env.transition(reward=reward, observation=observation)

#     def observation_spec(self):
#         return specs.Array(shape=(1, 2), dtype=np.float32, name="observation")

#     def action_spec(self):
#         return specs.DiscreteArray(self._n_actions, name="action")

#     def _save(self, observation):
#         self._raw_observation = (observation * 255).astype(np.uint8)

#     @property
#     def optimal_return(self):
#         # Returns the maximum total reward achievable in an episode.
#         return 1.1

#     def bsuite_info(self) -> Dict[str, Any]:
#         return {}

# """Simple diagnostic memory challenge.

# Observation is given by n+1 pixels: (context, time_to_live).

# Context will only be nonzero in the first step, when it will be +1 or -1 iid
# by component. All actions take no effect until time_to_live=0, then the agent
# must repeat the observations that it saw bit-by-bit.
# """


# class MemoryChain(base.Environment):
#     """Memory Chain environment, implementing the environment API."""

#     def __init__(
#         self, memory_length: int, num_bits: int = 1, seed: Optional[int] = None
#     ):
#         """Builds the memory chain environment."""
#         super(MemoryChain, self).__init__()
#         self._memory_length = memory_length
#         self._num_bits = num_bits
#         self._rng = np.random.RandomState(seed)

#         # Contextual information per episode
#         self._timestep = 0
#         self._context = self._rng.binomial(1, 0.5, num_bits)
#         self._query = self._rng.randint(num_bits)

#         # Logging info
#         self._total_perfect = 0
#         self._total_regret = 0
#         self._episode_mistakes = 0

#         # bsuite experiment length.
#         self.bsuite_num_episodes = 10_000  # Overridden by experiment load().

#     def _get_observation(self):
#         """Observation of form [time, query, num_bits of context]."""
#         obs = np.zeros(shape=(1, self._num_bits + 2), dtype=np.float32)
#         # Show the time, on every step.
#         obs[0, 0] = 1 - self._timestep / self._memory_length
#         # Show the query, on the last step
#         if self._timestep == self._memory_length - 1:
#             obs[0, 1] = self._query
#         # Show the context, on the first step
#         if self._timestep == 0:
#             obs[0, 2:] = 2 * self._context - 1
#         return obs

#     def _step(self, action: int) -> dm_env.TimeStep:
#         observation = self._get_observation()
#         self._timestep += 1

#         if self._timestep - 1 < self._memory_length:
#             # On all but the last step provide a reward of 0.
#             return dm_env.transition(reward=0.0, observation=observation)
#         if self._timestep - 1 > self._memory_length:
#             raise RuntimeError("Invalid state.")  # We shouldn't get here.

#         if action == self._context[self._query]:
#             reward = 1.0
#             self._total_perfect += 1
#         else:
#             reward = -1.0
#             self._total_regret += 2.0
#         return dm_env.termination(reward=reward, observation=observation)

#     def _reset(self) -> dm_env.TimeStep:
#         self._timestep = 0
#         self._episode_mistakes = 0
#         self._context = self._rng.binomial(1, 0.5, self._num_bits)
#         self._query = self._rng.randint(self._num_bits)
#         observation = self._get_observation()
#         return dm_env.restart(observation)

#     def observation_spec(self):
#         return specs.Array(
#             shape=(1, self._num_bits + 2), dtype=np.float32, name="observation"
#         )

#     def action_spec(self):
#         return specs.DiscreteArray(2, name="action")

#     def _save(self, observation):
#         self._raw_observation = (observation * 255).astype(np.uint8)

#     def bsuite_info(self):
#         return dict(total_perfect=self._total_perfect, total_regret=self._total_regret)


# class BsuiteWrapper(gym.Env):
#     """
#     This class wraps bsuite enviroments
#     Available Environments: 
#     * memory_len/0
#     * discounting_chain/0
#     """
#     metadata = {'render.modes': ['human', 'rgb_array']}
#     def __init__(self, env_name, N, reset_params = None, realtime_mode = False):
#         """Instantiates the bsuite environment.
        
#         Arguments:
#             env_name {string} -- Name of the bsuite environment
#             reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
#             realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
#         Attributes: 
#             observation_space_shape {tuple} -- Shape of the observation space
            
#         """

#         if env_name == 'MemoryLength':
#             self._env = MemoryChain(N)
#         elif env_name == 'DiscountingChain':
#             self._env = DiscountingChain(N)

#         # self._env = bsuite.load_from_id(env_name)  # type: dm_env.Environment
#         self._env.reset()
#         self.max_episode_steps = self._env._episode_len


#         # self._env.seed(self._default_reset_params['start-seed'])

#         self._last_observation = None  # type: Optional[np.ndarray]
#         self.viewer = None
#         self.game_over = False  # Needed for Dopamine agents.

#         obs_spec = self._env.observation_spec()  # type: specs.Array
#         if isinstance(obs_spec, specs.BoundedArray):
#             self.observation_space = spaces.Box(
#                 low=float(obs_spec.minimum),
#                 high=float(obs_spec.maximum),
#                 shape=obs_spec.shape,
#                 dtype=obs_spec.dtype)
#         self.observation_space = spaces.Box(
#             low=-float('inf'),
#             high=float('inf'),
#             shape=obs_spec.shape,
#             dtype=obs_spec.dtype)

#         self._rewards = []
#         self.t = 0

#         if reset_params is None:
#             self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
#         else:
#             self._default_reset_params = reset_params

#         # TO-DO: add another envs
#         # if you want to use another popgym enviroment just add desired rows below

#         if isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
#             if len(self.observation_space.shape) > 2:
#                 self.observation_space.obs_type = 'image'
#                 self.observation_space.obs_shape = self.observation_space.shape # fix!
#             else:
#                 self.observation_space.obs_type = 'vector'
#                 self.observation_space.obs_shape = (self.observation_space.shape[-1],)
#         elif isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
#             # discrete observation
#             self.observation_space.obs_shape = (1, )
#             self.observation_space.obs_type = 'discrete'
#         elif isinstance(self.observation_space, Iterable) and isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
#             # multi discrete observation - like discrete vector 
#             self.observation_space.obs_shape = (1 for obs_space in env.observation_space )
#             self.observation_space.obs_type = 'iterable_discrete'
#         elif isinstance(self.observation_space, Iterable) and isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
#             # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
#             raise NotImplementedError

#     # @property
#     # def max_episode_steps(self):
#     #     """Returns the maximum number of steps that an episode can last."""



#     def step(self, action): # Tuple[np.ndarray, float, bool, Dict[str, Any]]
#         """Runs one timestep of the environment's dynamics.
        
#         Arguments:
#             action {list} -- The to be executed action
        
#         Returns:
#             {numpy.ndarray} -- Visual observation
#             {float} -- (Total) Scalar reward signaled by the environment
#             {bool} -- Whether the episode of the environment terminated
#             {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
#         """

#         if isinstance(action, list):
#             if len(action) == 1:
#                 action = action[0]

#         timestep = self._env.step(action)
#         if timestep.last():
#             self.game_over = True
#         self._last_observation = timestep.observation

#         reward = timestep.reward or 0.
#         obs, terminated, info = timestep.observation, self.game_over, {}
#         # print(tuple(reversed(obs.shape)))
#         self._rewards.append(reward)

#         if isinstance(obs, int):
#             obs = np.array([obs, ]) #.reshape((self.observation_space.obs_shape))
#         elif isinstance(obs, Iterable):
#             obs = obs.flatten() #.reshape((self.observation_space.obs_shape))
#         if self.t == self.max_episode_steps - 1:
#             terminated = True


#         if terminated:
#             info = {"reward": sum(self._rewards),
#                     "length": len(self._rewards)}

#         self.t += 1

#         return obs, reward, terminated, info

#     def reset(self) -> np.ndarray:
#         """Resets the environment.
        
#         Keyword Arguments:
#             reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
#         Returns:
#             {numpy.ndarray} -- Visual observation
#         """
#         self.t = 0
#         self._rewards = []
#         self.game_over = False

#         # Reset the environment to retrieve the initial observation
#         timestep = self._env.reset()
#         # self._env.seed(self._default_reset_params['start-seed'])

#         self._last_observation = timestep.observation
#         obs = timestep.observation


#         if isinstance(obs, int):
#             obs = np.array([obs, ])
#         elif isinstance(obs, Iterable):
#             obs = obs.flatten()
#         return obs

#     def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
#         if self._last_observation is None:
#           raise ValueError('Environment not ready to render. Call reset() first.')

#         if mode == 'rgb_array':
#           return self._last_observation

#         if mode == 'human':
#           if self.viewer is None:
#             # pylint: disable=import-outside-toplevel
#             # pylint: disable=g-import-not-at-top
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.SimpleImageViewer()
#           self.viewer.imshow(self._last_observation)
#           return self.viewer.isopen


#     @property
#     def action_space(self) -> spaces.Discrete:
#         action_spec = self._env.action_spec()  # type: specs.DiscreteArray
#         return spaces.Discrete(action_spec.num_values)

#     @property
#     def reward_range(self) -> Tuple[float, float]:
#         reward_spec = self._env.reward_spec()
#         if isinstance(reward_spec, specs.BoundedArray):
#           return reward_spec.minimum, reward_spec.maximum
#         return -float('inf'), float('inf')

from environments.Bsuite.bsuite_env import BsuiteWrapper

# # # env =  # memory_len/0
env = BsuiteWrapper('MemoryLength', (4, 3)) #MemoryLength
# # # print(env.action_space.n)
# # # print(env.observation_space)
# # # print(env.observation_space.obs_type)

# # # env = BsuiteWrapper('DiscountingChain')

# # # env = BsuiteWrapper('memory_len/10')
# # #env = BsuiteWrapper('discounting_chain/3')
# # # env = gym_wrapper.GymFromDMEnv(bsuite.load_from_id('discounting_chain/4'))
# # # print(env.action_space, type(env.action_space))
# # # print(env.observation_space, type(env.observation_space))
# # # print(env.__dict__)
# # # print(env._env.bsuite_num_episodes)

# # # a = env.reset()
# # # env.seed(42)
# # # print(a)
# # # env.observation_space.kkkk = 444env.observation_space
# # # env.observation_space.name = 'sdfsdf'

# # # setattr(env.observation_space, 'name','nams')


# # # env.observation_space.name = 'sdfsd'
# # # print(env.observation_space.__dict__)
# # # print(env.observation_space.name)

# # # print(env.observation_space.obs_shape)
# # # print(env.observation_space.obs_type)

for i in range(5):
    print(f'step {i}')
    action = env.action_space.sample()
    if i == 29:
        action = 3
    obs, reward, terminated, info  = env.step(action)
    print(action)
    if reward != 0.0:
        print(f'STEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {i}')
    print(obs, obs.shape, reward, terminated, info)
    #print(action)



# # # from bsuite import sweep

# # # print(sweep.SWEEP)



# # # print(sweep.DISCOUNTING_CHAIN)

# # from bsuite.environments import memory_chain
# # from bsuite.experiments.memory_len import sweep

# # print(sweep.NUM_EPISODES)








# # print(sweep.NUM_EPISODES)
# # print(env.observation_space.obs_type)
# # print(env.observation_space.obs_shape)


### TEST DiscountingChain GAME 


# from absl.testing import absltest
# from bsuite.environments import discounting_chain
# from dm_env import test_utils

# import numpy as np


# env = discounting_chain.DiscountingChain(4)
# valid_actions = [0, 1, 2, 3, 4]
# rng = np.random.RandomState(42)
# reward_sum = 0

# for i in range(101):
#     action = rng.choice(valid_actions)
#     # print(action)
#     if i == 1:
#         res = env.step(4)
#     else:
#         res = env.step(0)
#     reward_sum += res.reward or 0
#     if res.reward != 0:
#         print(i, res)

# print(reward_sum)


### TEST MemoryLength GAME 

# from bsuite.environments import memory_chain

# env = memory_chain.MemoryChain(memory_length=4, num_bits=3)

# valid_actions = [0, 1]
# rng = np.random.RandomState(42)

# reward_sum = 0
# init_obs = None

# for i in range(6):
#     print(f'step {i}')
#     action = rng.choice(valid_actions)
#     # print(action)
#     action = 0
#     if i == 5:
#         print(res.observation)
#         action = init_obs[int(res.observation[0][1]) + 2]
#     res = env.step(action)
#     print(f'action: {action}')
#     if i == 0:
#         init_obs = res.observation[0]
#     reward = res.reward or 0
#     reward_sum += reward
#     print(res)
#     if reward != 0:
#         print(f'reward on step {i} = {reward}')

# print(reward_sum)
