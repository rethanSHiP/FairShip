#!/bin/bash

ENERGY = $1
DEST_DIR= $2

cd /afs/cern.ch/work/r/rethan/public/FairShip
sleep $((RANDOM % 30))
source /cvmfs/ship.cern.ch/26.04/setUp.sh

mkdir -p ${DEST_DIR}

alienv setenv FairShip/latest -c python macro/run_simScript.py -n 1000 --tag "muon_${ENERGY}GeV" -o Simulations/${DEST_DIR} PG --pID 13 --Estart ${ENERGY} --Eend ${ENERGY} --Vz 3200
