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

import inspect
from statistics import mean, stdev
import subprocess

from tqdm import tqdm 

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

from agents.keep_agent import load_pretrained
from agents.inductive_keep_agent import load_inductive_pretrained
from graph_wrapper.wrapper import GraphWrapper


MAX_EPS = 100
SAVE_ACTION_DISTRO = False
agent_name = 'Blue'

'''
Copied from CybORG directory 
'''

def wrap(env):
    return GraphWrapper('Blue', env)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    # ask for a name
    name = 'Isaiah, Benjamin, & Howie' 
    # ask for a team
    team = 'Cybermonic' 
    # ask for a name for the agent
    name_of_agent = 'KEEP'

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]
    
    # Loading a pretrained graph PPO agent
    agent = load_pretrained() # Non-inductive model 
    # agent = load_inductive_pretrained() # Inductive model
    agent.set_deterministic(True)

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    '''
    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")
    '''
        
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
            
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})

            wrapped_cyborg = wrap(cyborg)
            observation = wrapped_cyborg.reset()

            total_reward = []
            actions = []
            for i in tqdm(range(MAX_EPS), desc=str(red_agent), total=MAX_EPS):
                r = []
                a = []
                
                for j in range(num_steps):
                    action = agent.get_action(observation)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    
                    #action_log[red_agent][str(cyborg.get_last_action('Blue'))] += 1
                    true_state = cyborg.get_agent_state('True')

                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation

                observation = wrapped_cyborg.reset()

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            
            '''
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
            '''