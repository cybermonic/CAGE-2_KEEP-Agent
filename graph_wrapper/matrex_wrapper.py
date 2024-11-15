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
import json

import torch
from CybORG.Shared.Actions import *

from graph_wrapper.observation_graph import ObservationGraph
import graph_wrapper.utils as utils


class GraphWrapper:
    # In order of trained model expected output
    ACTIONS = [
        Analyse, Remove,
        DecoyApache, DecoyFemitter, DecoyHarakaSMPT, DecoySmss,
        DecoySSHD, DecoySvchost, DecoyTomcat, DecoyVsftpd,
        Restore,
    ]
    HOSTNAMES = [] # Filled in by self.reset()

    def __init__(self, env, action_space, version, write_obs=False):
        # Need to keep the original environment before wrapping it
        self.original_env = env

        if version == 'v5':
            crown_jewels = ['Op_Auth', 'Op_Database']
        else:
            crown_jewels = ['Op_Server']

        self.graph = ObservationGraph(crown_jewels=crown_jewels)
        self.crown_jewels = crown_jewels
        self.table = None
        self.write_obs = write_obs

        self.env = env
        self.possible_actions = []
        self.action_space = action_space
        self.action_mapping = utils.build_action_map_dict(action_space, self.ACTIONS)

        self.step_counter = 0
        self.last_obs = None

    def step(self, action=None):
        self.step_counter += 1

        # NOTE: return type is now dict instead of tuple
        action_obj = Sleep() if action is None else self.to_action_object(action)
        result_dict = self.env.step(action_obj)

        if self.write_obs:
            out_dict = {k:result_dict[k] for k in ['reward', 'observation']}
            out_dict['action'] = str(action)
            with open(f'obs_{self.step_counter}.json', 'w+') as f:
                f.write(json.dumps(out_dict, default=lambda x : str(x), indent=1))

        dict_obs = result_dict['observation']

        # Tell tabular what happened and update
        tabular_obs = self.table.observation_change(dict_obs, action_obj)

        # Tell ObservationGraph what happened and update
        self.graph.parse_observation(action_obj, dict_obs)

        # Get graph data/update the state of the graph
        x,ei = self.graph.get_state()

        # Combine tabular and graph data
        x = self._combine_data(x, tabular_obs)
        self.last_obs = (x,ei)

        # Again, caution, expected return type is now dict
        result_dict['observation'] = (x,ei)
        return result_dict

    def reset(self, obs=None):
        self.step_counter = 0

        # Reset tabular data
        if obs is None:
            obs = self.env.reset()['observation']

        if self.write_obs:
            out_dict = {'observation': {k:obs[k] for k in obs.keys()}}
            out_dict['action'] = 'N/a'
            out_dict['reward'] = 'N/a'

            with open(f'{self.write_obs}-obs_{self.step_counter}.json', 'w+') as f:
                f.write(json.dumps(out_dict, default=lambda x : str(x), indent=1))

        self.table = utils.BlueTable(init_obs = obs)
        tab_x = self.table.observation_change(obs, None)

        # Reset graph
        self.graph = ObservationGraph(crown_jewels=self.crown_jewels)
        self.graph.setup(obs)
        graph_x, ei = self.graph.get_state()

        # Combine observations
        x = self._combine_data(graph_x, tab_x)
        self.last_obs = (x,ei)

        # Get new topology (if needed)
        self.HOSTNAMES = self.graph.hostnames
        action_space = self.env.action_mapping({"agent": "Blue"})
        self.action_mapping = utils.build_action_map_dict(action_space, self.ACTIONS)

        # Again, need to change return type to dict
        return {
            'observation': (x, ei)
        }

    def _combine_data(self, graph_x, tabular_x):
        tabular_x = tabular_x.reshape(tabular_x.shape[0]//4, 4)

        additional_data = torch.zeros(graph_x.size(0), 4)
        additional_data[graph_x[:,0] == 1] = torch.tensor(tabular_x, dtype=torch.float32)
        return torch.cat([graph_x, additional_data], dim=1)

    def to_action_object(self, action):
        if action is None:
            return None

        target = self.HOSTNAMES[action // len(self.ACTIONS)]
        act = self.ACTIONS[action % len(self.ACTIONS)]
        return act(hostname=target, agent='Blue', session=0)

    def action_translator(self, action):
        '''
        Unlike previous wrapper, assumes actions are indexed by
        action id rather than node id.
            E.g. actions 0,1,2 are [A_1(n_1), A_2(n_1), A_3(n_1)]
                               NOT [A_1(n_1), A_1(n_2), A_1(n_3)]

        Need to flip this back around before reusing the old code
        '''
        if action is None:
            return None

        target = self.HOSTNAMES[action // len(self.ACTIONS)]
        act = self.ACTIONS[action % len(self.ACTIONS)]
        act_id = self.action_mapping[act][target]

        return act_id