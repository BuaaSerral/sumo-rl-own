# DQN

## Single Agent

### How to act once

```python
class dqn_agent:
    def __init__(self, observation_space, action_space, exploration_strategy= dqn_epsilon_greedy, gamma=0.95) -> None:
        self.exploration_strategy = dqn_epsilon_greedy(self)

    def act(self,state) -> int:#Change global action!
    """Choose action based on value_model."""
    self.action = self.exploration.choose(agent= self, state=state)
    return self.action

class dqn_epsilon_greedy(epsilon_greedy):
    """dqn Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon = 0.05, min_epsilon = 0.01, decay = 0.99)
        pass

    def choose(self, agent, state) -> int:
        """
        Choose action based on epsilon greedy strategy.
        agent: class: dqn_agent
        """
        if np.random.rand() < self.epsilon:
            action = int(agent.action_space.sample())
        else:
            action = agent.value_model(torch.FloatTensor(state).reshape(1, agent.observation_dims)).argmax().item()
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        return action
```

### How to play once

```python
class dqn_pool:
    '''dqn_pool class'''
    def __init__(self, env, agent: dqn_agent) -> None:
        '''env: SumoEnvironment(gym.Env)'''
        pass

    def play_once(self) -> Tuple[list,float]:
        data = []
        reward_sum = 0.0
        done = {'__all__': False}
        state, _ = self.env.reset()
        while not done['__all__']:
            actions = self.agent.act(state)
            next_state, reward, _, done['__all__'], __ = self.env.step(actions)
            data.append((state, actions, reward, next_state, done))
            reward_sum += reward
            state = next_state
        return data, reward_sum

class SumoEnvironment(gym.Env):
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if self.fixed_ts or action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

class TrafficSignal:
    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
```

### Pooling

With```sumo_rl\nets\2way-single-intersection\single-intersection-gen.rou.xml```,\
$$
env_{steps} = 1e5, \delta = 5 \\
data_{len} = 1e5 / 5 = 2e4 \\
$$
It implies that if play the whole env, we'll get 2e4 data.However, in actually train, it'll cost too much time. So i change the route file, consider 3600s.\
$$
env_{steps} = 3600, \delta = 5 \\
data_{len} = 3600 / 5 = 720 \\
$$

```python
class dqn_pool:
    def update(self):
        '''
        every time update, data should be > 1440 \n
        only save the newest 1440 data
        '''
        old_len = len(self.pool)
        while len(self.pool) - old_len < 1440 - 1:
            self.pool.extend(self.play_once()[0])
        self.pool = self.pool[-1440:]
```

Note dqn_pool.sample, it should be consistent with the hidden layer.

```python
class dqn_pool:
    def sample(self):
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
```

### Dependence

---

.sumo_rl.agents.dqn_agent.dqn_agent
.sumo_rl.agents.dqn_agent.dqn_pool
.sumo_rl.exploration.epsilon_greedy.dqn_epsilon_greedy

---
