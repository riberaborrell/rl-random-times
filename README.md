# rl-for-gym
This Python repository contains the implementation of reinforcement learning algorithms for the gymansium classic conrol and MuJoCo environments.

## Contains
- Policy gradient (PG) with (random time horizon) trajectories expectations and q-value estimated by the returns (REINFORCE algorithm for stochastic policies).
- Policy gradient (PG) with on-policy expectations with Z-factor estimated (unbiased) and with Z-factor neglected (biased).

## Install

1. clone the repo 
```
git clone git@github.com:riberaborrell/rl-for-gym.git
```

2. move inside the directory, create virtual environment and install required packages
```
cd rl-for-gym
make venv
```

3. activate venv
```
source venv/bin/activate
```

4. create config.py file and edit it
```
cp src/rl_for_gym/config_template.py src/rl_for_gym/config.py
```

## Usage
### Algorithms
1. PG with random time horizon trajectories (batch version of the original REINFORCE algorithm for random time horizons)
```
$ python src/rl_for_gym/spg/run_reinforce_stoch.py --expectation-type random-time --return-type initial-return
```

2. PG with on-policy expectations unbiased (z-factor estimated)
```
$ python src/rl_for_gym/spg/run_reinforce_stoch.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1 --estimate-z
```

3. PG with on-policy expectations biased (z-factor neglected)
```
$ python src/rl_for_gym/spg/run_reinforce_stoch.py --expectation-type on-policy --return-type n-return --mini-batch-size 0.1
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

* on-policy expectation: the algorithm samples K trajectories and stores them (the experiences i.e. states, action and reward tuple) in a memory. The on-policy expectation samples from this memory. If K_type = 'adaptive' (default), K_mini represents the percentage of data in the memory. Else, K_type = 'constant', K_mini represents the absolute number of experiences sampled.
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
