# import os
# import sys
# sys.path.append(r'C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl')
# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy

# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")
# import traci

# from sumo_rl import SumoEnvironment


# if __name__ == "__main__":
#     env = SumoEnvironment(
#         net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
#         route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
#         out_csv_name="outputs/2way-single-intersection/dqn",
#         single_agent=True,
#         use_gui=False,
#         num_seconds=1e5,
#     )

#     model = DQN.load("./model/dqn.pkl")
#     mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
#     env.close()
#     print(mean_reward, std_reward)
    
class abc:
    def _init_(self) -> None:
        pass
    def __call__(self) -> int:
        return 1
t = abc()
print(f'My value = {t()}')

#(sumo-rl) PS C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl> python .\experiments\temp.py
#My value = 1