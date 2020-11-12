python generate.py m2s-v0 configs/m2s_env.json 2000 ./data/gen_m2s
python pretrain_visual_cortex.py --config ./configs/pretrain_full.json --env m2s-v0 --env-config ./configs/m2s_env.json --env-data-dir=./data/gen_m2s --env-obs-key=full --model-file=./data/pretrain/full.pt --epochs 7
python train_stub_agent.py m2s-v0 configs/m2s_env.json configs/stub_agent_env_full.json configs/stub_agent_full.json
