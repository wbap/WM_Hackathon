
cd ../../cerenaut-pt-core
python setup.py develop

cd ../wm_hackathon/env/

# export SDL_VIDEODRIVER="x11"
# echo " +++++++++++++++++++++++ SDL_VIDEODRIVER = $SDL_VIDEODRIVER"
cmd='python keyboard_agent.py dm2s-v0 configs/dm2s_env.par'
echo $cmd
eval $cmd