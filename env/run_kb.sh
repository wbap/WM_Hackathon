
cd ../../cerenaut-pt-core
python setup.py develop
cd ../WM_Hackathon/env

# export SDL_VIDEODRIVER="x11"
# echo " +++++++++++++++++++++++ SDL_VIDEODRIVER = $SDL_VIDEODRIVER"
python keyboard_agent.py dm2s-v0 configs/dm2s_env.par