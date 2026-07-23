#!/bin/bash

ENERGY=$1
DEST_DIR=$2

WORKDIR=/afs/cern.ch/work/r/rethan/public/FairShip
cd ${WORKDIR}
sleep $((RANDOM % 30))
source /cvmfs/ship.cern.ch/26.04/setUp.sh

OUTPATH=${WORKDIR}/Simulations/${DEST_DIR}
mkdir -p ${OUTPATH}

mkdir -p /afs/cern.ch/work/r/rethan/public/FairShip/${DEST_DIR}
alienv setenv FairShip/latest -c python macro/run_simScript.py -n 1000 --tag "muon_${ENERGY}GeV" \
-o ${OUTPATH} --shieldName TRY_2025 \
PG --pID 13 --Estart ${ENERGY} --Eend ${ENERGY} --Vz 3200 --Vy 200 --thetaMin -2.29 --thetaMax -2.29
