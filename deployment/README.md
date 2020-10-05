
1. Set up Docker
2. Build the container `bash build-docker.sh`
3. Change to the base folder: `cd /env`
4. Run the script `run_kb.sh` in the container: `bash run-docker.sh run_kb.sh` 

Note that you can run anything in the container, but we need to use run_kb.sh because it installs an additional dependency (cerenaut-pt-core)
