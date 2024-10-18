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
from CybORG.Shared.Actions import Restore, Remove, Action

from agents.variable_topology_agent import load_agent
from agents.inductive_keep_agent import load_inductive_pretrained
from graph_wrapper.wrapper import InductiveGraphWrapper


MAX_EPS = 100
SAVE_ACTION_DISTRO = False
agent_name = 'Blue'


FP = 0; TP = 1
def calc_precision(env: CybORG, last_act: Action):
    '''
    Returns (Action_string, int) where Action_string is either "Remove" or "Restore"
    and the int is [0,1] denoting if that action was a true positive (e.g. a restore
    on a rooted host) or a false positive (e.g. restore on a clean/user-level access host)

    Could do some extra metrics about restoring on a user level host. Is that really a false
    positive? Unclear. Restoring a clean host for sure is an FP though...
    '''
    def determine_red_access(session_list):
        '''
        Stolen from TrueTableWrapper
        '''
        for session in session_list:
            if session['Agent'] != 'Red':
                continue
            privileged = session['Username'] in {'root','SYSTEM'}
            return 'Root' if privileged else 'User'

        return 'None'

    ret = None

    # As of right now, only calculating TP/FP on Restore and Remove
    if isinstance(last_act, Restore) or isinstance(last_act, Remove):
        target = last_act.hostname
        true_state = env.get_agent_state('True')[target]['Sessions']
        red_foothold = determine_red_access(true_state)

        last_act_str = str(last_act).split(' ')[0]

        if red_foothold == 'None':
            ret = FP
        elif red_foothold == 'User':
            ret = FP if last_act_str == 'Restore' else TP # TP only if user-level shell was `Remove`d
        elif red_foothold == 'Root':
            ret = TP if last_act_str == 'Restore' else FP # TP only if root-level shell was `Restore`d

        ret = (last_act_str, ret)

    return ret

'''
Copied from CybORG directory
'''
def wrap(env):
    return InductiveGraphWrapper('Blue', env)

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
    agent = load_agent('model_weights/inductive_agent.pt') # Default rewards model
    # agent = load_agent('model_weights/high_precision.pt') # Reward-shaping model
    agent.set_deterministic(True)

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

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
            precision = {'Restore': [], 'Remove': []}
            for i in tqdm(range(MAX_EPS), desc=str(red_agent), total=MAX_EPS):
                r = []
                a = []

                for j in range(num_steps):
                    action = agent.get_action(observation)
                    pr = calc_precision(cyborg, wrapped_cyborg.to_action_object(action))
                    observation, rew, done, info = wrapped_cyborg.step(action)

                    if pr is not None:
                        k,v = pr
                        precision[k].append(v)

                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation

                observation = wrapped_cyborg.reset()

            restore = precision['Restore']
            if restore:
                pr_restore = sum(restore) / len(restore)
            else:
                pr_restore = 'NaN'

            remove = precision['Remove']
            if remove:
                pr_remove = sum(remove) / len(remove)
            else:
                pr_remove = 'NaN'

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} +/- {stdev(total_reward)}')
            print(f'Restore precision: {pr_restore}\tRemove precision: {pr_remove}')