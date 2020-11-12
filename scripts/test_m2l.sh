python generate.py m2s-v0 configs/m2s_av_env.json 2000 ./data/gen_m2s_av

#Pre-train fovea
python pretrain_visual_cortex.py --config ./configs/pretrain_fovea.json --env m2s-v0 --env-config ./configs/m2s_av_env.json --env-data-dir=./data/gen_m2s_av --env-obs-key=fovea --model-file=./data/pretrain_m2s_av/fovea.pt --epochs 7

#Pre-train peripheral
python pretrain_visual_cortex.py --config ./configs/pretrain_peripheral.json --env m2s-v0 --env-config ./configs/m2s_av_env.json --env-data-dir=./data/gen_m2s_av --env-obs-key=peripheral --model-file=./data/pretrain_m2s_av/peripheral.pt --epochs 7

#Run move_to_light game
python train_stub_agent.py m2l-v0 configs/m2l_av_env.json configs/stub_agent_env_av.json configs/stub_agent_av.json
