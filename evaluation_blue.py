# NOTE this is the evaluation file for evaluating our Agent using
# the Matrex API. This file needs to be installed along with Matrex API and CybORG
import sys
sys.path.append('/root') # if you run in docker container

from MaTrExApi import *
from Agents import *
from statistics import mean, stdev

from tqdm import tqdm
from CybORG.Shared.Actions import *

from load_agents import *
from graph_wrapper.matrex_wrapper import GraphWrapper

MAX_EPS = 100

if __name__ == '__main__':
    for num_steps in [30, 50, 100]:
        for red_agent in ["B_lineAgent", "RedMeanderAgent", "SleepAgent"]:
            # Create a new session
            session = MaTrExApi()

            request = {
                        "client_name": "KRYPTOWIRE",
                        "MaTrEx_version": "MaTrEx_v1"
                        }

            session_info = session.new_session(request)
            print(f"Evaluating Cybermonic GraphPPO agent against {red_agent} with {num_steps} steps for {MAX_EPS} episodes.")


            # Specify game parameters.
            new_game_params = {
                                "client_token": session_info["client_token"],
                                "scenario": "Scenario2.yaml",
                                "main_agent": "Blue",
                                "blue_agent_0": "play",
                                "red_agent_0": red_agent,
                                "green_agent_0": "SleepAgent",
                                "wrapper": None,
                                "episode_length": num_steps,
                                "max_episodes": MAX_EPS+1,
                                "seed": 0
                                }
            # Initiate a new game with requested parameters.
            game_info = session.new_game(new_game_params)

            # new_game() returns information about action space and observation
            if game_info["status"] == "success":
                action_space = game_info["action_space"]
                obs = game_info["observation"]
                action_mapping_dict = game_info["action_mapping_dict"]
            else:
                print(f"Game initiation failed with the following message:\n{game_info}")
                print()


            # Initiate your agent(action_space, observation)
            agent = load_global_attn_inductive_agent()
            total_reward = []

            # During the game, you can request action_mapping for a specified agent name, as shown in the example below.
            blue_action_mapping = session.action_mapping({"agent": "Blue"})

            # Wrapping in Kryptowire GraphWrapper to handle action/state mapping
            # and tracking internal graph edits
            wrapped = GraphWrapper(session, blue_action_mapping)

            # Need to reset env before playing
            observation = wrapped.reset()['observation']

            # Start the game loop for the specified number of episodes MAX_EPS and steps per episode EPS_LEN.
            total_reward = []
            for eps in tqdm(range(MAX_EPS), desc='Processing'):
                r = []

                for steps in range(num_steps):
                    # Get action based on current obsarvation with agent.get_action(observation, action_space)
                    action = agent.get_action(observation)

                    # Send that action with step() function
                    result = wrapped.step(action)

                    # Calling session.step() returns a dictionary containing information about observation, reward, done, info, action_space, and action_mapping_dict (if action_mapping is set to True in the step() call).
                    if result["status"] == "success":
                        observation = result["observation"]
                        reward = result["reward"]
                        done = result["done"]
                        info = result["info"]
                        action_space = result["action_space"]
                        action_mapping_dict = result["action_mapping_dict"]

                    else:
                        print(f"Step failed with the following message:\n{result}")
                        print()

                    r.append(reward)

                # Reset environment at the end of the episode
                reset_result = wrapped.reset()
                observation = reset_result["observation"]

                agent.end_episode()
                total_reward.append(sum(r))

            # Terminate your session at the end of the game. Specify your unique client_token for this session, which can be found in session_info as shown below.
            terminate_request = {"client_token": session_info["client_token"]}
            response = session.terminate(terminate_request)

            # Print results of the terminate() request.
            print(f"Session terminated with:\n{response}")
            print()

            # Request logs for the game. If successful, your logs will be saved in your working directory in the ./kafka-logs/{session_id} folder as {session_id}.json file.
            response = session.get_logs({"client_token": session_info["client_token"]})

            # Print results of the get_logs() request.
            print(f"get_logs() returned:\n{response}")
            print()

            print(f'Average reward for red agent {red_agent} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')