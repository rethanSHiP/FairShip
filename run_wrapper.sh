#!/bin/bash
cd /afs/cern.ch/work/r/rethan/public/FairShip
sleep $((RANDOM % 30))
source /cvmfs/ship.cern.ch/26.04/setUp.sh

alienv setenv FairShip/latest -c python macro/run_simScript.py -n 1000 --tag "muon_${1}GeV" -o Simulations/energy_scan PG --pID 13 --Estart $1 --Eend $1 --Vz 3200
