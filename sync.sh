#!/bin/bash

################################################################################
# sync code
################################################################################

export MACHINE=${1:-incbox}
DEST_DIR='~/agief-remote-run/WM_Hackathon'

# sync this folder
cmd="rsync --chmod=ug=rwX,o=rX --perms -av ./ $MACHINE:$DEST_DIR --exclude='.git/' --filter=':- .gitignore'"
echo $cmd
eval $cmd
status=$?

if [ $status -ne 0 ]
then
  echo "ERROR:  Could not complete rsync operation - failed at 'sync this folder' stage." >&2
  echo "	Error status = $status" >&2
  echo "	Exiting now." >&2
  exit $status
fi
