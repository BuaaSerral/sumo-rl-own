# MDP

## Observe

### Observe_definition

Observation occurs when it is time to act, and the key method```_compute_observation()``` depends on ```Trafficsignal.compute_observation()```.

```python
class SumoEnvironment(gym.Env):
    def __init__(observation_class: ObservationFunction = DefaultObservationFunction) -> None:
        self.observation_class = observation_class
    def step(self, action: Union[dict, int]):
        observations = self._compute_observations()
        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    def _compute_observations(self):
        self.observations.update(
            {
                ts: self.traffic_signals[ts].compute_observation()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {
            ts: self.observations[ts].copy()
            for ts in self.observations.keys()
            if self.traffic_signals[ts].time_to_act or self.fixed_ts
        }
```

Register observation_class by method ```_init_()```.\
```.observation.ObservationFunction._call_()``` will be used when ```compute_observation()``` is used in ```SumoEnvironment._compute_observations()```.\
Warning: ```_observation_fn_default()``` is meaningless, which is equal to ```.observation.DefaultObservationFunction().__call__()```

```python
class TrafficSignal:
    def __init__(**kwargs):
        self.observation_fn = self.env.observation_class(self)
        self.observation_space = self.observation_fn.observation_space()
    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()
    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation
```

```ObservationFunction._call_()``` is used in ```TrafficSignal.compute_observation()``` because ```self.observation_fn``` was registered in ```TrafficSignal._init_()``` and first called as ```self.observation_fn()```.

```python
class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
```

Actually,

```python
DefaultObservation = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
```

For example, consider a 2way-single-intersection,

```python
observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
phase_id=[1, 0, 0, 0]
min_green=[0]
density=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#2 way
queue=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#2 way
observation.shape = (21,)#4 + 1 + 8 + 8
```

### Observe_fetch

For single agent:

```python
class SumoEnvironment(gym.Env):
    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
class TrafficSignal:
    def __init__(**kwargs):
        self.observation_fn = self.env.observation_class(self)
        self.observation_space = self.observation_fn.observation_space()
class DefaultObservationFunction(ObservationFunction):
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
```

According to ```env.observation_space```, we get a ```gymnasium.spaces.Box``` as return.\
Finally, the dims of observation_space is equal to ```env.observation_space.shape[0]```.

```python
class Box(Space[NDArray[Any]]):
@property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape
```

## Action

### Action_definition

Acction in sumo-rl is to set the next green phase for the traffic signals.\
If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)\
If multiagent, actions is a dict {ts_id : greenPhase}

```python
class SumoEnvironment(gym.Env):
    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].action_space
    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.traffic_signals[ts_id].action_space
```

### Action_fetch

```python
class Discrete(Space[np.int64]):
    def sample(self, mask: MaskNDArray | None = None) -> np.int64
```

## Reward

```python
class  SumoEnvironment(gym.Env):
    def __init__(
        self, reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
    ) -> None:
        self.reward_fn = reward_fn
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    conn,
                )
                for ts in self.reward_fn.keys()#For different reward_fn
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    conn,
                )
                for ts in self.ts_ids
            }    
class TrafficSignal:
    def __init__(
        self,
        reward_fn: Union[str, Callable],#Callable refers to a function
    ):
        self.reward_fn = reward_fn
        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")
    
    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }
```
