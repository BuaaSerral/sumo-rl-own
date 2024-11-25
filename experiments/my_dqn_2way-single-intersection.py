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
        net_file=r'sumo_rl\nets\2way-single-intersection\single-intersection.net.xml',
        route_file=r'sumo_rl\nets\2way-single-intersection\single-intersection-gen.rou.xml',
        out_csv_name=r'outputs\2way-single-intersection\my_own_dqn',
        single_agent=True,
        use_gui=False,
        num_seconds=3600,
    )

from sumo_rl.agents.dqn_agent import dqn_agent,dqn_pool
my_agent = dqn_agent(env.observation_space, env.action_space)
my_pool = dqn_pool(env,my_agent)
my_agent.train(my_pool)

env.close()