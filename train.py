import sys
from statistics import mean, stdev
from types import SimpleNamespace

from threading import Thread
import torch
from tqdm import tqdm

from CybORG.Shared.Actions import *
from MaTrExApi import *

from load_agents import load_global_attn_inductive_agent
from graph_wrapper.matrex_wrapper import GraphWrapper
from agents.utils import PPOMemory
from agents.inductive_keep_agent import InductiveGraphPPOAgent

EP_LEN = 100    # How long episdes should be
N_EPS = 10      # How many episodes per training update

HYPERPARAMS = SimpleNamespace(
    batch_size=2048,
    epochs=20_000,          # How many times to generate episodes and .learn()
    learn_epochs = 4        # How many times we update during agent.learn()
)

def init_session(red_agent):
    session = MaTrExApi()
    request = {
        "client_name": "KRYPTOWIRE",
        "MaTrEx_version": "MaTrEx_v1"
    }
    session_info = session.new_session(request)

    # Specify game parameters.
    new_game_params = {
        "client_token": session_info["client_token"],
        "scenario": "Scenario2.yaml",
        "main_agent": "Blue",
        "blue_agent_0": "play",
        "red_agent_0": red_agent,
        "green_agent_0": "SleepAgent",
        "wrapper": None,
        "episode_length": EP_LEN,
        "max_episodes": N_EPS+1,
        "seed": None # Want randomness during training
    }

    # Initiate a new game with requested parameters.
    game_info = session.new_game(new_game_params)

    # new_game() returns information about action space and observation
    if game_info["status"] != "success":
        print(f"Game initiation failed with the following message:\n{game_info}")
        print()
        return None

    blue_action_mapping = session.action_mapping({"agent": "Blue"})
    wrapped = GraphWrapper(session, blue_action_mapping)

    return wrapped, session, session_info

@torch.no_grad()
def generate_episodes(i, agent, red_agent):
    env, session, session_info = init_session(red_agent)
    buff = PPOMemory(1)

    # Error message is printed in the init_session method of
    # this occurs. Just return None to gracefully exit
    if env is None:
        return buff

    for _ in tqdm(range(N_EPS), desc=str(i)):
        observation = env.reset()['observation']
        for step in range(EP_LEN):
            action, value, prob = agent.get_action(observation)
            result = env.step(action)

            if result["status"] == "success":
                next_observation = result["observation"]
                reward = result["reward"]
            else:
                print(f"Step failed with the following message:\n{result}")
                print()
                return buff

            buff.remember(observation, action, value, prob, reward, step==EP_LEN-1)
            observation = next_observation

    # This takes so fucking long, just start it up in it's own thread and
    # forget about it. It will eventually close. It usually takes about the same
    # amount of time as it does to do a full backprop with the data the fn returns
    # This saves about 10 seconds per epoch
    terminate_request = {"client_token": session_info["client_token"]}
    t = Thread(target=session.terminate, args=(terminate_request,))
    t.start()

    return buff

def train(hp, agent):
    log = {'e':[], 'r':[]}
    red_agents = ["B_lineAgent", "RedMeanderAgent", "Kryptowire_DQN_Red", "Kryptowire_DQN_Reduced_Red", "RedPILLS_Cypher0_Red"]
    n_red = len(red_agents)

    for e in range(1,hp.epochs+1):
        # Generate N_EPS episodes against each possible red agent
        buffs = Parallel(n_jobs=len(red_agents), prefer='processes')(
            delayed(generate_episodes)(i, agent, red_agent) for i,red_agent in enumerate(red_agents)
        )

        # Sum of each memory buffer's reward area avg'd for N_EPS episodes,
        # across each red agent
        avg_r = mean([sum(b.r) / N_EPS for b in buffs])
        print(f'[{e}] Avg r: {avg_r}')
        log['e'].append(e*n_red*N_EPS)
        log['r'].append(avg_r)

        # Combine output of all episode generation, and update policy
        agent.memory = PPOMemory(hp.batch_size).combine(buffs)
        agent.learn()

        # Checkpoint and log
        agent.save(outf=f'checkpoints/{hp.name}_checkpoint.pt')
        torch.save(log, f'logs/{hp.name}.pt')
        if e % 1000 == 0:
            torch.save(out_f=f'checkpoints/{hp.name}-{(e*n_red*N_EPS)//1000}k.pt')


from joblib import Parallel, delayed
def test():
    agent = load_global_attn_inductive_agent()
    agent.train()

    red_agents = ["B_lineAgent", "RedMeanderAgent", "Kryptowire_DQN_Red", "Kryptowire_DQN_Reduced_Red", "RedPILLS_Cypher0_Red"]
    buffs = Parallel(n_jobs=25, prefer='processes')(
        delayed(generate_episodes)(i, agent, red_agent) for i,red_agent in enumerate(red_agents)
    )

    agent.memory = PPOMemory(2048).combine(buffs)
    agent.learn()


if __name__ == '__main__':
    # Start session so we can get input dimension for model
    env, session, session_info = init_session('SleepAgent')
    x,ei = env.reset()['observation']
    dim = x.size(1)

    # Kill session
    terminate_request = {"client_token": session_info["client_token"]}
    t = Thread(target=session.terminate, args=(terminate_request,))
    t.start()

    # Train fresh globalnode agent
    agent = InductiveGraphPPOAgent(
        dim,
        bs=HYPERPARAMS.batch_size,
        a_kwargs={'lr': 0.0003},
        c_kwargs={'lr': 0.001},
        clip=0.2,
        epochs=HYPERPARAMS.learn_epochs,
        globalnode=True
    )

    HYPERPARAMS.name = 'globalnode_v2'
    train(HYPERPARAMS, agent)