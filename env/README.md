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

## Match to Sample task
`python keyboard_agent.py m2s-v0 configs/m2s_env.par `

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

### Match to Sample
`python generate.py m2s-v0 configs/m2s_env.par 2000 ./data/gen`

### Delayed Match to Sample
`python generate.py dm2s-v0 configs/dm2s_env.par 2000 ./data/gen`

## Pre-training modules
Example:

`python pretrain_visual_cortex.py --config ./configs/pretrain_fovea.json --env dm2s-v0 --env-config ./configs/dm2s_env.par --env-data-dir=./data/gen --env-obs-key=fovea --model-file=./data/pretrain/fovea.pt --epochs 10`

To view the output of pretraining, you can examine the tensorboard output in the ./run directory.

To start tensorboard in this folder, use a command such as:

`tensorboard --logdir=. --port=6008 --samples_per_plugin images=200`

## Training the RL agent
This actually trains the Reinforcement-Learning parts of the Agent on the task. It reloads pretrained networks for posterior cortex and other brain modules that are not trained rapidly or via RL.

`python train_stub_agent.py TASK_ENV TASK_ENV_CONFIG_FILE MODEL_CONFIG_FILE AGENT_CONFIG_FILE`

Example:

`python train_stub_agent.py dm2s-v0 configs/dm2s_env.par configs/stub_model.json configs/stub_agent.json`

Note that MODEL_CONFIG_FILE should configure and reload pretrained networks in Cortex. 

AGENT_CONFIG_FILE should configure the RL agent network.

# Stub Validation 
This section describes the steps that can be performed to validate the functionality of the provided stubs, and that the architecture as a whole can solve tasks.

1. `python keyboard_agent.py m2s-v0 configs/m2s_env.par` 
2. `python generate.py m2s-v0 configs/m2s_env.par 2000 ./data/gen_m2s`
3. `python pretrain_visual_cortex.py --config ./configs/pretrain_full.json --env m2s-v0 --env-config ./configs/m2s_env.par --env-data-dir=./data/gen_m2s --env-obs-key=full --model-file=./data/pretrain/full.pt --epochs 7`
4. `python train_stub_agent.py m2s-v0 configs/m2s_env.par configs/stub_model_full.json configs/stub_agent.json`