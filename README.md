# rl-random-times
This Python repository contains the implementation of reinforcement learning algorithms with random time horizons.

## Contains
- (random-time) trajectory-based policy gradient (*trajectory PG*) with q-value estimated by the returns (REINFORCE algorithm with random time horizons).
- (on-policy) state-space policy gradient (*state-space PG*) with with Z-factor estimated (unbiased) and with Z-factor neglected (biased).
- (random-time) trajectory and model-based policy gradient for deterministic policies (*model-based trajectory DPG*). The value function is estimated by the returns (REINFORCE algorithm with random time horizons and deterministic policies).
- (on-policy) state-space and model-based policy gradient for deterministic policies (*model-based, state-space DPG*) with with Z-factor estimated (unbiased) and with Z-factor neglected (biased).

## Install

1. clone the repo 
```
git clone git@github.com:riberaborrell/rl-random-times.git
```

2. move inside the directory, create virtual environment and install required packages
```
cd rl-random-times
make venv
```

3. activate venv
```
source venv/bin/activate
```

4. create config.py file and edit it
```
cp src/rl_random_times/config_template.py src/rl_random_times/config.py
```

## Dependencies
1. clone the gym-sde-is repo
```
cd ../
git clone git@github.com:riberaborrell/gym-sde-is.git
```
2. Pip install the gym-sde-is repo locally
```
pip install -e gym-sde-is
```

## Usage
### Algorithms
#### Stochastic policies
1. Trajectory PG
```
$ python src/rl_random_times/spg/run_reinforce_stoch.py --expectation-type random-time --return-type initial-return
```

2.  State-space PG unbiased (i.e. z-factor is estimated).
```
$ python src/rl_random_times/spg/run_reinforce_stoch.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1 --estimate-z
```

3. State-space PG biased (i.e. z-factor neglected).
```
$ python src/rl_random_times/spg/run_reinforce_stoch.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1
```
#### Deterministic policies
4. Model-based trajectory DPG for the gym-sde-is environment
```
$ python src/rl_random_times/dpg/run_modelbased_dpg_sdeis_dwnd.py
 --expectation-type random-time --return-type initial-return
```

5. Model-based and State-space DPG unbiased (i.e. z-factor is estimated).

```
$ python src/rl_random_times/dpg/run_modelbased_dpg_sdeis_dwnd.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1 --estimate-z
```


6. Model-based and State-space DPG biased (i.e. z-factor is neglected).
```
$ python src/rl_random_times/dpg/run_modelbased_dpg_sdeis_dwnd.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1
```

### Parameters
* Environment, maximal number of steps and seed: choose environment supported by the Envpool repository 
	1. Classic Control (https://envpool.readthedocs.io/en/latest/env/classic_control.html)
	2. Mujoco (https://envpool.readthedocs.io/en/latest/env/mujoco_gym.html)

```
$ python path/to/algorithm --env-id InvertedDoublePendulum-v4  --n-steps-lim 1000 --seed 1
```

* NN architecture: choose initial covariance of the Gaussian stochastic policy, if the covariance is learnt or constant, number of layers and size of hidden layers 
```
$ python path/to/algorithm --policy-noise 1.0 --gaussian-policy-type learnt-cov --n-layers 3 --d-hidden 32
```

* optimization: choose type of optimizer, learning rate, batch-size and number of gradient iterations
```
$ python path/to/algorithm --optim-type adam --batch-size 100 --lr 0.001 --n-grad-iterations 1000 
```

* (on-policy) state-space expectation: the algorithm samples K trajectories and stores them (the experiences i.e. states, action and reward tuples for each time step) in a memory. The (on-policy) state-space expectation samples from this memory. If K_type = 'adaptive' (default), K_mini represents the percentage of data in the memory. Else, K_type = 'constant', K_mini represents the absolute number of experiences sampled.
```
$ python path/to/algorithm --mini-batch-size-type adaptive --mini-batch-size 0.2 
```

* log and backup results frequency:
```
$ python path/to/algorithm --log-freq 100 --backup-freq 1000
```

* load run algorithm and plot results:
```
$ python path/to/algorithm --load --plot 
```
