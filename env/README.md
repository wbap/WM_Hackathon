# Install custom Gym environment:
You need to pre-install the custom Gym environments contained in this repository.

```
cd WM_Hackathon/env
pip install --ignore-installed --no-deps -e .
```

# Human agent unit tests
These tests allow you to evaluate custom env gameplay manually.

## Simple RL task
`python keyboard_agent.py simple-v0 configs/simple_env_human.json`

## Delayed Match to Sample task
`python keyboard_agent.py dm2s-v0 configs/dm2s_env.par `


# Reinforcement Learning agent test
These tests allow you to verify that the RL training regime works.

## Simple RL task
`python train_simple_agent.py simple-v0 configs/simple_env_machine.json configs/simple_agent_model.json`

Example output:
` 71 reward   0.00/  1.87/  4.00 len 389.30 saved tmp/simple/checkpoint_71/checkpoint-71`

The rewards are min/mean/max per epoch.
Should optimize to around 0/4/5.

## Delayed Match to Sample task
This allows you to run the modular agent with stubs on the DM2S task.

`python train_stub_agent.py dm2s-v0 configs/dm2s_env.par configs/simple_agent_model.json`

# Pretraining phase
You can pretrain parts of a model on a custom environment without rewards. This has two parts: First, pre-generate some data from the environment. Second, pretrain model modules on this data.

## Pre-generating data
Example:

`python generate.py dm2s-v0 configs/dm2s_env.par 100 dm2s.pickle configs/simple_model.json`

## Pre-training modules
Example:

`python pretrain.py --config ../../../cfsl/tests/test_configs/sae.json --env dm2s-v0 --env-config configs/dm2s_env.par --env-data-file=data.pickle --env-obs-key=fovea --epochs 3`

To view the output of pretraining, you can examine the tensorboard output in the ./run directory.

To start tensorboard in this folder, use a command such as:

`tensorboard --logdir=. --port=6008 --samples_per_plugin images=200`

