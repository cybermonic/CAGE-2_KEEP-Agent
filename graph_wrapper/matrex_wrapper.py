# Cybermonic CASTLE KEEP Agent
# Copyright (C) 2024 Cybermonic LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from Agents.KryptowireAgents.Utility import utils
from CybORG.Shared.Actions.Action import Sleep

from ..graph_wrapper.observation_graph import ObservationGraph

class GraphWrapper:
    def __init__(self, env, action_space):
        # Need to keep the original environment before wrapping it
        self.original_env = env
        self.graph = ObservationGraph()
        self.table = None

        self.env = env
        self.possible_actions = []

        # Minus 1 because one key is 'success': True
        for i in range(len(action_space)-1):
            self.possible_actions.append(
                utils.map_action(i, action_space)
            )

        self.step_counter = None
        self.last_obs = None

    def step(self, action=None):
        # NOTE: return type is now dict instead of tuple
        action = Sleep() if action is None else self.possible_actions[action]
        result_dict = self.env.step(action)
        dict_obs = result_dict['observation']

        # Tell tabular what happened and update
        tabular_obs = self.table.observation_change(dict_obs, action)

        # Tell ObservationGraph what happened and update
        self.graph.parse_observation(action, dict_obs)

        # Get graph data/update the state of the graph
        x,ei = self.graph.get_state()

        # Combine tabular and graph data
        x = self._combine_data(x, tabular_obs)
        self.last_obs = (x,ei)

        # Again, caution, expected return type is now dict
        result_dict['observation'] = (x,ei)
        return result_dict

    def reset(self):
        # Reset tabular data
        obs = self.env.reset()['observation']
        self.table = utils.BlueTable(init_obs = obs)
        tab_x = self.table.observation_change(obs, None)

        # Reset graph
        self.graph = ObservationGraph()
        self.graph.setup(obs)
        graph_x, ei = self.graph.get_state()

        # Combine observations
        x = self._combine_data(graph_x, tab_x)
        self.last_obs = (x,ei)

        # Again, need to change return type to dict
        return {
            'observation': (x, ei)
        }

    def _combine_data(self, graph_x, tabular_x):
        tabular_x = tabular_x.reshape(tabular_x.shape[0]//4, 4)

        additional_data = torch.zeros(graph_x.size(0), 4)
        additional_data[graph_x[:,0] == 1] = torch.tensor(tabular_x, dtype=torch.float32)
        return torch.cat([graph_x, additional_data], dim=1)