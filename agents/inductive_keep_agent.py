import torch 
from torch import nn 
from torch.optim import Adam 
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv

from graph_wrapper.observation_graph import ObservationGraph
from agents.keep_agent import GraphPPOAgent, PPOMemory, N_HOSTS, N_HOST_ACTIONS, N_GLOBAL_ACTIONS

DEFAULT_DIM = ObservationGraph.DIM + 4 

class SelfAttention(nn.Module):
    def __init__(self, in_dim, enc_dim, heads):
        super().__init__()

        self.q = nn.Linear(in_dim, enc_dim, bias=False)
        self.k = nn.Linear(in_dim, enc_dim, bias=False)
        self.v = nn.Linear(in_dim, enc_dim, bias=False)
        self.out = nn.MultiheadAttention(enc_dim, heads, batch_first=True)

    def forward(self, x):
        '''
        Expects B x L x d input
        '''
        q = self.q(x); k = self.k(x); v = self.v(x) 
        return self.out(q,k,v, need_weights=False)[0]
    

class SelfAttentionInductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes=N_HOSTS, action_space=N_HOST_ACTIONS, 
                 hidden1=256, hidden2=64, lr=0.0003):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            SelfAttention(hidden2, hidden1, 8),
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

        nbatches = host_z.size(0) // self.num_nodes 
        host_z = host_z.reshape(nbatches, self.num_nodes, host_z.size(1))
        
        actions = self.out(host_z)          # B x N x a
        actions = actions.transpose(1,2)    # B x a x N
        actions = actions.reshape(          # B x a*N
            nbatches, 
            self.num_nodes * self.action_space
        ) 

        out = self.sm(actions)
        out = torch.cat([torch.zeros(nbatches,N_GLOBAL_ACTIONS), out], dim=1)
        return Categorical(out)


class NaiveInductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes=N_HOSTS, action_space=N_HOST_ACTIONS, 
                 hidden1=256, hidden2=64, lr=0.0003):
        super().__init__()

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
        out = torch.cat([torch.zeros(nbatches,N_GLOBAL_ACTIONS), out], dim=1)
        return Categorical(out)
    

class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes=N_HOSTS, hidden1=256, 
                 hidden2=64, lr=0.001):
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

class InductiveGraphPPOAgent(GraphPPOAgent):
    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6, 
                 a_kwargs=dict(), c_kwargs=dict(), training=True, naive=True):
        if naive:
            self.actor = NaiveInductiveActorNetwork(in_dim, **a_kwargs)
        else:
            self.actor = SelfAttentionInductiveActorNetwork(in_dim, **a_kwargs)

        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)
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

def load_inductive_pretrained(in_f='model_weights/inductive_naive.pt', naive=True, num_nodes=None):
    data = torch.load(in_f)

    args,kwargs = data['agent']

    # Being used for an env with a different number of hosts
    if num_nodes:
        kwargs['a_kwargs']['num_nodes'] = num_nodes
        kwargs['c_kwargs']['num_nodes'] = num_nodes

    agent = InductiveGraphPPOAgent(*args, **kwargs, naive=naive)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])

    agent.eval()
    return agent