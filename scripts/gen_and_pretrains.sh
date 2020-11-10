# ------ m2s pretrain ------
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python generate.py m2s-v0 configs/m2s_env.json 2000 ./data/gen_m2s
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python pretrain_visual_cortex.py --config ./configs/pretrain_full.json --env m2s-v0 --env-config ./configs/m2s_env.json --env-data-dir=./data/gen_m2s --env-obs-key=full --model-file=./data/pretrain/full.pt --epochs 7
# ./data/pretrain/full.pt    <-------- ensure config points here
# sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python train_stub_agent.py m2s-v0 configs/m2s_env.json configs/stub_agent_env_full.json configs/stub_agent_full.json


# ------ m2s_av gen and pretrain ------
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python generate.py m2s-v0 configs/m2s_av_env.json 2000 ./data/gen_m2s_av
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python pretrain_visual_cortex.py --config ./configs/pretrain_fovea.json --env m2s-v0 --env-config ./configs/m2s_av_env.json --env-data-dir=./data/gen_m2s_av --env-obs-key=fovea --model-file=./data/pretrain_m2s_av/fovea.pt --epochs 7
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python pretrain_visual_cortex.py --config ./configs/pretrain_peripheral.json --env m2s-v0 --env-config ./configs/m2s_av_env.json --env-data-dir=./data/gen_m2s_av --env-obs-key=peripheral --model-file=./data/pretrain_m2s_av/peripheral.pt --epochs 7
# ./data/pretrain_m2s_av/    <---- ensure config points here

# m2l run
# sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python train_stub_agent.py m2l-v0 configs/m2l_av_env.json configs/stub_agent_env_av.json configs/stub_agent_av.json


# ------ dm2s generate and pretrain ------
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python generate.py dm2s-v0 configs/dm2s_env.json 2000 ./data/gen/dm2s
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python pretrain_visual_cortex.py --config ./configs/pretrain_fovea.json --env dm2s-v0 --env-config ./configs/dm2s_env.json --env-data-dir=./data/gen/dm2s --env-obs-key=fovea --model-file=./data/pretrain/fovea.pt --epochs 10
sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python pretrain_visual_cortex.py --config ./configs/pretrain_peripheral.json --env dm2s-v0 --env-config ./configs/dm2s_env.json --env-data-dir=./data/gen/dm2s --env-obs-key=peripheral --model-file=./data/pretrain/peripheral.pt --epochs 10
# sudo deployment/run-docker.sh ~/agief-remote-run/WM_Hackathon/ ~/agief-remote-run/cerenaut-pt-core/ true python train_stub_agent.py dm2s-v0 configs/dm2s_env.json configs/stub_agent_env_av.json configs/stub_agent_full.json


