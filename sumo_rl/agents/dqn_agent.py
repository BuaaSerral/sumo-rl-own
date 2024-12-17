"""DQN Agent class."""
from sumo_rl.exploration.epsilon_greedy import dqn_epsilon_greedy
from sumo_rl.environment.env import SumoEnvironment
import torch,random
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import os
class dqn_agent:
    """DQN Agent class."""

    def __init__(self, observation_space, action_space, exploration_strategy= dqn_epsilon_greedy, gamma=0.95) -> None:
        """Initialize DQN agent.\n
        observation_space: gymnasium.spaces.Box\n
        action_space: gym.spaces.Discrete
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.exploration_strategy = dqn_epsilon_greedy(self)
        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dims, 64),torch.nn.ReLU(),
            torch.nn.Linear(64, 64),torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dims)
        )
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dims, 64),torch.nn.ReLU(),
            torch.nn.Linear(64, 64),torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dims)
        )
        self.target_model.load_state_dict(self.value_model.state_dict())
        self.exploration = exploration_strategy()
        self.acc_reward = 0

    @property
    def observation_dims(self) -> int:
        return self.observation_space.shape[0]
    
    @property
    def action_dims(self) -> int:
        return self.action_space.n
        
    def act(self,state) -> int:#Change global action!
        """Choose action based on value_model."""
        self.action = self.exploration.choose(agent= self, state=state)
        return self.action

    def train(self, pool) -> None:
        self.value_model.train()
        optimizer = torch.optim.Adam(self.value_model.parameters(), lr=2e-4) # type: ignore
        loss_fn = torch.nn.MSELoss()
        for epoch in range(1_00):
            print(f'\ncurrent epoch = {epoch}')
            pool.update()
            print('\nPool update over')
            for k in range(1_00):
                state, action, reward, next_state, over = pool.sample()
                value = self.value_model(state).gather(dim=1, index=action)
                #state_action_values = policy_net(state_batch).gather(1, action_batch)
                #计算target
                with torch.no_grad():
                    target = self.target_model(next_state)
                target = target.max(dim=1)[0].reshape(-1, 1)
                #target.max(dim=1)转换为torch.return_types.max类型的元组，0为数值，1为索引
                #reshape方法将tensor中所有数值转换为列向量
                target = target * 0.99 * (1 - over) + reward
                loss = loss_fn(value, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            #copy
            if (epoch + 1) % 5 == 0:
                self.target_model.load_state_dict(self.value_model.state_dict())
            #eval
            if (epoch + 1) % 10 == 0:
            #This step will slow speed,because we call the function play_once() which will use sumo
                test_result = sum([pool.play_once()[-1] for _ in range(2)]) / 2
                print(f'\nepoch={epoch}, len={len(pool)},test_result={test_result}\n')
class dqn_mutli_agent():
    """DQN Mutli Agent class."""
    def __init__(self, env:SumoEnvironment) -> None:
        self.ts_ids = env.ts_ids
        self.agents = {
            ts_id : dqn_agent(env.observation_spaces(ts_id), env.action_spaces(ts_id))
            for ts_id in self.ts_ids
        }
        self.env = env
    @property
    def agent_dict(self) -> dict:
        return self.agents
    
    def train(self, pools) -> None:
        for ts_id in self.ts_ids:
            self.agents[ts_id].value_model.train()
        optimizers = {
            ts_id : torch.optim.Adam(self.agents[ts_id].value_model.parameters(), lr=2e-4)
            for ts_id in self.ts_ids
        }
        loss_fns = {
            ts_id : torch.nn.MSELoss()
            for ts_id in self.ts_ids
        }
        #dqn是off policy，那我池化一次就够了？
        pools.update()
        loss_list = {ts_id : [] for ts_id in self.ts_ids}
        for epoch in range(1_00):
            #pools.update()
            loss_avg = {ts_id : [] for ts_id in self.ts_ids}
            for _ in range(1_00):
                sample_data = pools.sample()
                for ts_id in self.ts_ids:
                    obs,action,reward,next_obs = sample_data[ts_id]
                    value = self.agents[ts_id].value_model(obs).gather(dim=1, index=action)
                    with torch.no_grad():
                        target = self.agents[ts_id].target_model(next_obs)
                    target = target.max(dim=1)[0].reshape(-1, 1)
                    target = target * 0.99 + reward
                    loss = loss_fns[ts_id](value,target)
                    loss_avg[ts_id].append(loss.item())
                    loss.backward()
                    optimizers[ts_id].step()
                    optimizers[ts_id].zero_grad()
            for ts_id in self.ts_ids:
                loss_list[ts_id].append(np.mean(np.array(loss_avg[ts_id])))
            #copy
            if (epoch + 1) % 5 == 0:
                for ts_id in self.ts_ids:
                    self.agents[ts_id].target_model.load_state_dict(self.agents[ts_id].value_model.state_dict())
        #loss
        plt.figure()
        iter = 0
        for key,value in loss_list.items():
            plt.subplot(2,2,iter + 1)
            plt.plot(value, label=f'Intersection_{key}')
            plt.legend()
            iter += 1
        plt.tight_layout()  # 调整子图布局以避免重叠
        plt.show()    
            
            #eval
            # if (epoch + 1) % 20 == 0:
            # #This step will slow speed,because we call the function play_once() which will use sumo
            #     self.eval(epoch)
    def save(self) -> None:
        '''Hint: u r in ./experiments/ma_test.py'''
        for ts_id in self.ts_ids:
            value = f'models/ma_test/agent_{ts_id}_value_model.pth'
            target = f'models/ma_test/agent_{ts_id}_target_model.pth'
            torch.save(self.agents[ts_id].value_model,value)
            torch.save(self.agents[ts_id].target_model,target)
    def load(self, location) -> None:
        '''This method will modify self.agents'''
        location_value = {ts_id : '' for ts_id in self.ts_ids}
        location_target = {ts_id : '' for ts_id in self.ts_ids}
        for filename in os.listdir(location):
            #like agent_1_target_model.pth
            parts = filename.split('_')
            agent_id = parts[1]
            model_type = parts[2]
            if model_type == 'value':
                location_value[agent_id] = filename
            else:
                location_target[agent_id] = filename
        
        for key in self.agents.keys():
            self.agents[key].value_model = torch.load('./models/ma_test/' + location_value[key])
            self.agents[key].target_model = torch.load('./models/ma_test/' + location_target[key])
    def eval(self, epoch) -> None:
        done = {"__all__": False}
        obs = self.env.reset()
        while not done['__all__']:
            actions = {ts_id: self.agents[ts_id].act(obs[ts_id]) for ts_id in self.ts_ids}
            next_obs, _, done, _ = self.env.step(actions)
            for ts_id in self.ts_ids:
                obs[ts_id] = next_obs[ts_id]
        self.env.save_csv(r'outputs\2x2grid\ma_new',epoch)
                        
class dqn_pool:
    '''dqn_pool class'''
    def __init__(self, env, agent: dqn_agent) -> None:
        '''env: SumoEnvironment(gym.Env)'''
        self.pool = []
        self.env = env
        self.agent = agent
    def __len__(self) -> int:
        return len(self.pool)
    def __getitem__(self, i) -> tuple:
        '''(state, action, reward, next_state, truncated)'''
        return self.pool[i]
    def play_once(self) -> Tuple[list,float]:
        data = []
        reward_sum = 0.0
        done = {'__all__': False}
        state, _ = self.env.reset()
        while not done['__all__']:
            # actions = {ts : self.agent.act(state) for ts in self.env.ts_ids}
            # next_state, reward, _, done, __ = self.env.step(actions)
            actions = self.agent.act(state)
            next_state, reward, _, done['__all__'], __ = self.env.step(actions)
            data.append((state, actions, reward, next_state, done))
            reward_sum += reward
            state = next_state
        return data, reward_sum
    def update(self) -> None:
        '''
        every time update, data should be > 1440 \n
        only save the newest 1440 data
        '''
        # old_len = len(self.pool)
        # while len(self.pool) - old_len < 2e4 - 1:
        #     self.pool.extend(self.play_once()[0])
        # self.pool = self.pool[-2_0000:]
        old_len = len(self.pool)
        while len(self.pool) - old_len < 720 - 1:
            self.pool.extend(self.play_once()[0])
        self.pool = self.pool[-720:]
        
    def sample(self) -> tuple:
        data = random.sample(self.pool, 64)
        #For speeding up, convert list to np.array() first
        state_array = np.array([i[0] for i in data])
        action_array = np.array([i[1] for i in data])
        reward_array = np.array([i[2] for i in data])
        next_state_array = np.array([i[3] for i in data])
        done_array = np.array([i[4]['__all__'] for i in data])
        state = torch.FloatTensor(state_array).reshape(-1, 21)#21=obs.dims
        action = torch.LongTensor(action_array).reshape(-1, 1)#1=act
        reward = torch.FloatTensor(reward_array).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state_array).reshape(-1, 21)
        done = torch.LongTensor(done_array).reshape(-1, 1)
        return state, action, reward, next_state, done

class dqn_mutli_pool:
    '''dqn_mutli_pool class'''
    def __init__(self, env, agents: dict) -> None:
        '''env: SumoEnvironment(gym.Env)'''
        self.env = env
        self.agents = agents
        self.pools = {
            agent_id : []
            for agent_id in self.agents.keys()
        }
        self.ts_ids = self.agents.keys()
    def play_once(self) -> Tuple[list,float]:
        data = {ts_id : [] for ts_id in self.agents.keys()}
        done = {"__all__": False}
        obs = self.env.reset()
        while not done['__all__']:
            # actions = {ts : self.agent.act(state) for ts in self.env.ts_ids}
            # next_state, reward, _, done, __ = self.env.step(actions)
            # actions = self.agent.act(state)
            # next_state, reward, _, done['__all__'], __ = self.env.step(actions)
            # data.append((state, actions, reward, next_state, done))
            # reward_sum += reward
            # state = next_state
            actions = {ts_id: self.agents[ts_id].act(obs[ts_id]) for ts_id in self.ts_ids}
            next_obs, reward, done, _ = self.env.step(actions)
            for key in data.keys():
                data[key].append((obs[key], actions[key], reward[key], next_obs[key]))
                obs[key] = next_obs[key]
        return data
    def update(self) -> None:
        # old_len = len(self.pool)
        # while len(self.pool) - old_len < 2e4 - 1:
        #     self.pool.extend(self.play_once()[0])
        # self.pool = self.pool[-2_0000:]
        temp_agent_id = list(self.ts_ids)[0]
        old_len = len(self.pools[temp_agent_id])
        while len(self.pools[temp_agent_id]) - old_len < 1440 - 1:
            data = self.play_once()
            for ts_id in self.ts_ids:
                self.pools[ts_id].extend(data[ts_id])
        for ts_id in self.ts_ids:
            self.pools[ts_id] = self.pools[ts_id][-1440:]
        
    def sample(self) -> dict:
        data = {ts_id : random.sample(self.pools[ts_id], 64) for ts_id in self.ts_ids}
        out_data = {}
        #For speeding up, convert list to np.array() first
        for ts_id in self.ts_ids:
            obs_array = np.array([i[0] for i in data[ts_id]])
            action_array = np.array([i[1] for i in data[ts_id]])
            reward_array = np.array([i[2] for i in data[ts_id]])
            next_obs_array = np.array([i[3] for i in data[ts_id]])
            
            obs = torch.FloatTensor(obs_array).reshape(-1, 21)#21=obs.dims
            action = torch.LongTensor(action_array).reshape(-1, 1)#1=act
            reward = torch.FloatTensor(reward_array).reshape(-1, 1)
            next_obs = torch.FloatTensor(next_obs_array).reshape(-1, 21)
            
            out_data[ts_id] = (obs,action,reward,next_obs)
        return out_data