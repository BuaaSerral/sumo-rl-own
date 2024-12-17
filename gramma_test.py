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

# 抽象类测试
# class abc:
#     def _init_(self) -> None:
#         pass
#     def __call__(self) -> int:
#         return 1
# t = abc()
# print(f'My value = {t()}')

#(sumo-rl) PS C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl> python .\experiments\temp.py
#My value = 1

# from typing import Callable
# class test:
#     def __init__(self,name) -> None:
#         self._name_ = name
#         print(self._name_)
    
#     @classmethod
#     def register(cls, fn: Callable):
#         print(fn())
        
# a = test('qqq')
# def hello() -> int:
#     return 1

# def num(t:dict):
#     print(1)
# t = {1:1,2:2,3:3}
# # for i in t.keys():
# #     print(i)
# print(list(t.keys())[0])

# temp = {'1':1,'2':2,'3':3}
# for key,value in temp:
#     print(key,value)

import os

# 使用 os.listdir() 列出当前目录下的所有文件和文件夹
# directory_path = r'.\models\ma_test'
# for filename in os.listdir(directory_path):
#     print(filename)
#     parts = filename.split('_')
#     agent_id = parts[1]
#     model_type = parts[2]
#     print(agent_id)
#     print(model_type)
ts_ids = ['1','2','5','6']
def load(location) -> None:
    '''This method will modify self.agents'''
    location_value = {ts_id : '' for ts_id in ts_ids}
    location_target = {ts_id : '' for ts_id in ts_ids}
    for filename in os.listdir(location):
        parts = filename.split('_')
        agent_id = parts[1]
        model_type = parts[2]
        if model_type == 'value':
            location_value[agent_id] = filename
        else:
            location_target[agent_id] = filename
    print(location_target)
load(r'.\models\ma_test')
