from functools import partial
import pretrained
# from smac.env import MultiAgentEnv, StarCraft2Env
from envs.multiagentenv import MultiAgentEnv
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) \
                if type(done) is list \
                else not done
            done = len(observation) * [True]
        return observation, reward, done, info

class FlatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.env = env
        self.observation_space = None
        dim = self.n_objects * 2 * 2 + self.n_agents * 4
        self.single_space = gym.spaces.Box(np.ones(dim)*-1, np.ones(dim)*100, dtype=np.float32)
        self.observation_space = [self.single_space for _ in range(self.n_agents)]

    def observation(self, obs):
        """
        Flattens the observation dictionary into a 1D numpy array

        Args:
        obs (dict): The observation dictionary.

        Returns:
        list: List of flattened observation array as np.ndarray
        """
        flattened_obs_all = []
        num_agents = len(obs)

        # Information for each agent
        for i in range(num_agents):
            flattened_obs = []
            agent_key = f'agent_{i}'
            agent_obs = obs[agent_key]

            # Self information
            flattened_obs.extend(agent_obs['self']['position'])
            flattened_obs.append(int(agent_obs['self']['picker']))
            flattened_obs.append(agent_obs['self']['carrying_object']
                                 if agent_obs['self']['carrying_object'] is not None else -1)

            # Other agents' information
            for other_agent in agent_obs['agents']:
                flattened_obs.extend(other_agent['position'])
                flattened_obs.append(int(other_agent['picker']))
                flattened_obs.append(
                    other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

            # Objects' information
            for obj in obs[agent_key]['objects']:
                flattened_obs.extend(obj['position'])

            # Goals' information
            for goal in obs[agent_key]['goals']:
                flattened_obs.extend(goal)
            flattened_obs = np.array(flattened_obs)
            flattened_obs_all.append(flattened_obs)

        return flattened_obs_all

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, seed, **kwargs):
        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        if 'macpp' in key:
            self._env = FlatObs(self._env)
        else:
            self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if type(reward) is list:
            reward = sum(reward)
        if type(done) is list:
            done = all(done)
        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if hasattr(self.original_env, 'state_size'):
            return self.original_env.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self, actions=None):
        if actions is None:
            self._env.render()
        else:
            self._env.render(actions=actions)

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
