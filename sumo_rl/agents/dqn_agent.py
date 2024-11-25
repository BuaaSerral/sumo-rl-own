"""DQN Agent class."""
from sumo_rl.exploration.epsilon_greedy import dqn_epsilon_greedy
import torch,random
import numpy as np
from typing import Tuple
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
    