# usage: run-docker.ps1 [path to wm code] [boolean for GPU or not] [command and params to run in container]
# e.g.  deployment/run-docker.ps1 ~/Dev/WM_Hackathon ~/Dev/cerenaut-pt-core $false python 
Param(
	[String]$WM_SRC_DIR = "~/Dev/WM_Hackathon",
	[String]$CORE_SRC_DIR = "~/Dev/cerenaut-pt-core",
	[Boolean]$GPU = $false
)

[Array]$SHIFTED_ARGS = $args

$WM_TGT_DIR = "/root/wm_hackathon"
$CORE_TGT_DIR = "/root/cerenaut-pt-core"
$IMG_NAME = "cerenaut/wbai_wm_hackathon:latest"

if ($GPU) {
	$GPU_STR = "--gpus all"
}

$NetIPAddress = Get-NetIPAddress | Where-Object {($_.AddressFamily -eq "IPv4") -and ($_.InterfaceAlias -eq "vEthernet (WSL)")}
$IP = $NetIPAddress.IPAddress

$cmd = "docker run --privileged -it --rm --name=wm $GPU_STR -e DISPLAY=${IP}:0 -p 6006:6006 -e XAUTHORITY=/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/.Xauthority:/.Xauthority -v ${WM_SRC_DIR}:${WM_TGT_DIR} -v ${CORE_SRC_DIR}:${CORE_TGT_DIR} ${IMG_NAME} bash deployment/setupcore_and_run.sh $SHIFTED_ARGS"

Write-Output $cmd
Invoke-Expression $cmd
