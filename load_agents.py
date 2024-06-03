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

import os
from agents.transductive_keep_agent import load_transductive_ppo
from agents.inductive_keep_agent import load_inductive_ppo


WEIGHT_DIR = os.path.join(os.path.dirname(__file__), 'model_weights')

'''
Agent factory functions.
Locates proper weights and spins up models in eval() mode.
'''

def load_naive_inductive_agent(num_nodes=None):
    weightfile = os.path.join(WEIGHT_DIR, 'inductive_simple.pt')
    return load_inductive_ppo(in_f=weightfile, naive=True, globalnode=False, num_nodes=num_nodes)

def load_global_attn_inductive_agent(num_nodes=None):
    weightfile = os.path.join(WEIGHT_DIR, 'inductive_global-attn.pt')
    return load_inductive_ppo(in_f=weightfile, naive=False, globalnode=True, num_nodes=num_nodes)

def load_transductive_agent(num_nodes=None):
    weightfile = os.path.join(WEIGHT_DIR, 'noninductive.pt')
    return load_transductive_agent(in_f=weightfile)