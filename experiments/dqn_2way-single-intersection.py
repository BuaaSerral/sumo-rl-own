import os
import sys
sys.path.append(r'C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=1e5,
        reward_fn='pressure'
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=5e-4,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
        device='cuda'
    )
    model.learn(total_timesteps=1e5)
    env.close()
    model.save(r"./model/dqn.pkl")
'''
rollout/ep_rew_mean（或ep_rew_mean）：这个指标表示每个回合（episode）的平均奖励。在强化学习中，回合是指智能体与环境进行交互的一个完整过程，从初始状态到终止状态的一系列动作和观测。
"ep_rew_mean"图表展示了在训练过程中，每个回合的平均奖励的变化情况。通过观察这个指标的变化，可以了解训练过程中智能体的学习进展和性能提升情况。
rollout/ep_len_mean（或ep_len_mean）：这个指标表示每个回合（episode）的平均长度。回合的长度是指智能体在一个回合中执行的动作数量或步数。
"ep_len_mean"图表展示了在训练过程中，每个回合的平均长度的变化情况。通过观察这个指标的变化，可以了解智能体在不同阶段的训练中是否出现了过早终止或者无法收敛的问题。
'''