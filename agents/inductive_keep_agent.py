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
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv

from agents.transductive_keep_agent import GraphPPOAgent, PPOMemory

class NaiveInductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes=13, action_space=11, n_global_actions=2,
                 hidden1=256, hidden2=64, lr=0.0003):
        super().__init__()
        self.N_GLOBAL_ACTIONS = n_global_actions

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, action_space)
        )

        self.sm = nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr)

        self.num_nodes = num_nodes
        self.action_space = action_space

    def forward(self, x, ei):
        hosts = x[:, 0] == 1

        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))
        host_z = x[hosts]
        actions = self.out(host_z)

        # Group by p(a) (currently grouped by node id)
        nbatches = actions.size(0) // self.num_nodes

        # B x N x a
        out = actions.reshape(nbatches, self.num_nodes, actions.size(1))
        out = out.transpose(1,2) # Make rows actions, and columns nodes
        out = out.reshape(nbatches, self.num_nodes * actions.size(1)) # Combine batches into individual rows
        out = self.sm(out)

        # Need to also give probs of global events. Fortunately, these are ignored, so we set to zero
        out = torch.cat([torch.zeros(nbatches, self.N_GLOBAL_ACTIONS), out], dim=1)
        return Categorical(out)

    def forward_unshaped(self, x, ei):
        hosts = x[:, 0] == 1

        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))
        host_z = x[hosts]
        actions = self.out(host_z)

        return actions


class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes=13,
                 hidden1=256, hidden2=64, lr=0.001):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

        self.opt = Adam(self.parameters(), lr)
        self.num_nodes = num_nodes

    def forward(self, x, ei):
        hosts = x[:, 0] == 1

        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))

        vals = self.out(x[hosts])
        nbatches = vals.size(0) // self.num_nodes

        # Array of value of each host at this state
        vals = vals.reshape(nbatches, vals.size(0) // nbatches)

        # Just add them all up
        return vals.sum(dim=1, keepdim=True)


class SimpleSelfAttention(nn.Module):
    '''
    Implimenting global-node self-attention from
        https://arxiv.org/pdf/2009.12462.pdf
    '''
    def __init__(self, in_dim, h_dim, g_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Softmax(dim=-1)
        )
        self.feat = nn.Linear(in_dim, h_dim)
        self.glb = nn.Linear(h_dim+g_dim, g_dim)

        self.g_dim = g_dim
        self.h_dim = h_dim

    def forward(self, v, g=None):
        '''
        Inputs:
            v: B x N x d tensor
            g: B x d tensor
        '''
        if g is None:
            g = torch.zeros((v.size(0), self.g_dim))

        att = self.att(v)               # B x N x h
        feat = self.feat(v)             # B x N x h
        out = (att*feat).sum(dim=1)     # B x h

        g_ = self.glb(torch.cat([out,g], dim=-1))  # B x g
        return g + g_                               # Short-circuit


class GlobalNodeInductiveActorNetwork(NaiveInductiveActorNetwork):
    def __init__(self, in_dim, num_nodes=13, action_space=11, n_global_actions=2,
                 hidden1=256, hidden2=64, lr=0.0003):
        super().__init__(in_dim, num_nodes, action_space, n_global_actions, hidden1, hidden2, lr)
        self.N_GLOBAL_ACTIONS = n_global_actions

        gdim = hidden1
        self.out = nn.Sequential(
            nn.Linear(hidden2+gdim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, action_space)
        )

        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        self.opt = Adam(self.parameters(), lr)

    def __reshape_hosts(self, x):
        nbatches = x.size(0) // self.num_nodes
        return x.reshape(nbatches, self.num_nodes, x.size(1))

    def forward(self, x, ei):
        hosts = x[:, 0] == 1

        g = self.g0_attn(self.__reshape_hosts(x[hosts]))
        x = torch.relu(self.conv1(x, ei))

        g = self.g1_attn(self.__reshape_hosts(x[hosts]), g=g) # B x h2
        x = torch.relu(self.conv2(x, ei))

        g = self.g2_attn(self.__reshape_hosts(x[hosts]), g=g)                        # B x h2
        g = g.repeat_interleave(self.num_nodes, 0)      # B*N x h2

        host_z = torch.cat([x[hosts], g], dim=1)        # B*N x h2*2
        actions = self.out(host_z)                      # B*N x a

        # Group by p(a) (currently grouped by node id)
        nbatches = actions.size(0) // self.num_nodes

        # B x N x a
        out = actions.reshape(nbatches, self.num_nodes, actions.size(1))
        out = out.transpose(1,2) # Make rows actions, and columns nodes
        out = out.reshape(nbatches, self.num_nodes * actions.size(1)) # Combine batches into individual rows
        out = self.sm(out)

        # Need to also give probs of global events. Fortunately, these are ignored, so we set to zero
        out = torch.cat([torch.zeros(nbatches, self.N_GLOBAL_ACTIONS), out], dim=1)
        return Categorical(out)


class GlobalNodeInductiveCriticNetwork(InductiveCriticNetwork):
    def __init__(self, in_dim, num_nodes=13,
                 hidden1=256, hidden2=64, lr=0.001):
        super().__init__(in_dim, num_nodes, hidden1, hidden2, lr)
        gdim = hidden1

        self.out = nn.Sequential(
            nn.Linear(gdim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

        # Changing to maxpooling to mirror paper
        self.conv1 = GCNConv(in_dim, hidden1, aggr='max')
        self.conv2 = GCNConv(hidden1, hidden2, aggr='max')

        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        self.opt = Adam(self.parameters(), lr)

    def __reshape_hosts(self, x):
        nbatches = x.size(0) // self.num_nodes
        return x.reshape(nbatches, self.num_nodes, x.size(1))

    def forward(self, x, ei):
        hosts = x[:, 0] == 1

        g = self.g0_attn(self.__reshape_hosts(x[hosts]))
        x = torch.relu(self.conv1(x, ei))   # B*N x h1

        h = self.__reshape_hosts(x[hosts])  # B x N x h1
        g = self.g1_attn(h, g=g)            # B x h2
        x = torch.relu(self.conv2(x, ei))   # B*N x h2

        h = self.__reshape_hosts(x[hosts])  # B x N x h2
        g = self.g2_attn(h, g=g)            # B x h2
        return self.out(g) # B x 1

class InductiveGraphPPOAgent(GraphPPOAgent):
    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6,
                 a_kwargs=dict(), c_kwargs=dict(), training=True, naive=True, globalnode=False):
        if naive and not globalnode:
            self.actor = NaiveInductiveActorNetwork(in_dim, **a_kwargs)
            self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)
        elif globalnode:
            self.actor = GlobalNodeInductiveActorNetwork(in_dim, **a_kwargs)
            self.critic = GlobalNodeInductiveCriticNetwork(in_dim, **c_kwargs)
        else:
            raise NotImplementedError("We should never arrive at this part of the code")

        self.memory = PPOMemory(bs)

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training
        )

        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss()

def load_inductive_ppo(in_f='saved_models/inductive_ppo.pt', naive=True, num_nodes=None, globalnode=False):
    data = torch.load(in_f)

    args,kwargs = data['agent']

    # Being used for an env with a different number of hosts
    if num_nodes:
        kwargs['a_kwargs']['num_nodes'] = num_nodes
        kwargs['c_kwargs']['num_nodes'] = num_nodes

    agent = InductiveGraphPPOAgent(*args, **kwargs, naive=naive, globalnode=globalnode)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])

    agent.eval()
    return agent
