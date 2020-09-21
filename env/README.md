# Install custom Gym environment:
You need to pre-install the custom Gym environments contained in this repository.

```
cd WM_Hackathon/env
pip install  --ignore-installed --no-deps -e .
```

# Human agent unit tests
These tests allow you to evaluate custom env gameplay manually.

## Simple RL task
`python keyboard_agent.py simple-v0 simple_env_human.json`

## Delayed Match to Sample task
`python keyboard_agent.py dm2s-v0 ../games/dm2s/DM2S.par `


# Reinforcement Learning agent test
These tests allow you to verify that the RL training regime works.

## Simple RL task
`python train_simple_agent.py simple-v0 simple_env_machine.json simple_agent_model.json`

Example output:
` 71 reward   0.00/  1.87/  4.00 len 389.30 saved tmp/simple/checkpoint_71/checkpoint-71`

The rewards are min/mean/max per epoch.
Should optimize to around 0/4/5.

## Delayed Match to Sample task
This allows you to run the modular agent with stubs on the DM2S task.

`python train_stub_agent.py dm2s-v0 ../games/dm2s/DM2S.par simple_agent_model.json`