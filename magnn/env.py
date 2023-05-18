
import numpy as np
import pygame
from gymnasium.utils import EzPickle
from gymnasium.spaces import Discrete, Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated
from pettingzoo import AECEnv
from pettingzoo.sisl.pursuit.manual_policy import ManualPolicy
from pettingzoo.sisl.pursuit.pursuit_base import Pursuit as _env
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from magnn.transforms import obs_to_graph
from collections import OrderedDict

__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


def env(as_graph, **kwargs):
    env = raw_env(**kwargs)

    env.as_graph = as_graph
    if as_graph:
        env.observation_spaces = dict(zip(env.agents, [env.graph_observation_space] * env.num_agents))
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pursuit_v4",
        "is_parallelizable": True,
        "render_fps": 5,
        "has_manual_policy": True,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs)
        self.render_mode = kwargs.get("render_mode")
        self.as_graph = False
        pygame.init()
        self.agents = ["pursuer_" + str(a) for a in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.n_act_agents = self.env.act_dims[0]
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.graph_observation_space = Dict(
            {
                "node_features": Repeated(Box(low=0, high=7, shape=(3,), dtype=int), max_len=147),
                "edge_index": Repeated(Box(low=0, high=147, shape=(2,), dtype=int), max_len=294),
            }
        )

        self.steps = 0
        self.closed = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.env.reset()

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.close()

    def render(self):
        if not self.closed:
            return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        for k in self.agents:
            self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
        self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        o = self.env.safely_observe(self.agent_name_mapping[agent])
        o = np.swapaxes(o, 2, 0)
        if self.as_graph:
            edge_index, node_features = obs_to_graph(o, self.env.num_agents)
            o = OrderedDict([("edge_index", edge_index), ("node_features", node_features)])
           
        return o
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]