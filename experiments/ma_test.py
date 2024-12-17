# System Configuration check
import os
import sys
sys.path.append(r'C:\Users\win11\OneDrive\paper\Accumulation\sumo_learning_new\sumo-rl')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("'Please declare the environment variable 'SUMO_HOME'")

# env init
from sumo_rl import SumoEnvironment
env = SumoEnvironment(
        net_file=r'sumo_rl\nets\2x2grid\2x2.net.xml',
        route_file=r'sumo_rl\nets\2x2grid\2x2.rou.xml',
        out_csv_name=r'outputs\2x2grid\ma_test',
        use_gui=False,
        num_seconds=3600,
        reward_fn='queue'
    )
from sumo_rl.agents.dqn_agent import dqn_agent,dqn_mutli_pool,dqn_mutli_agent

# 训练
my = dqn_mutli_agent(env)
pools = dqn_mutli_pool(env,my.agent_dict)
# my.train(pools)
# my.save()
# fixed
# done = {"__all__": False}
# env.reset()
# while not done["__all__"]:
#     _, _, done, _ = env.step(None)
# env.save_csv(r"outputs\2x2grid\fixed",1)
# print(my.agent_dict['1'].value_model.state_dict()['0.weight'])
# print('------------------------------------------')
my.load(r'.\models\ma_test')
# print('------------------------------------------')
# print(my.agent_dict['1'].value_model.state_dict()['0.weight'])
my.eval(4)
env.close()
