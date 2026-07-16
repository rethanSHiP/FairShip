import argparse
import os
import glob
import uproot
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import awkward as ak

# Point this to the correct geometry file for your simulation
geo_file = "energy_scan/geo_muon_50GeV.root" 
# This populates ROOT.gGeoManager
ROOT.TGeoManager.Import(geo_file)

import TrackExtrapolateTool


def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--field_file", default="../files/2025_02_12_SHiP_SpectrometerField_ECN3_MgB2.root", help="Path to magnet field file")
    parser.add_argument("--sim_file", default="energy_scan/sim_*.root", help="Path to simulation files")
    parser.add_argument("--z_center", type=float, default=8957.0, help="Z offset for field map")
    parser.add_argument("--UBT_z", type=float, default=3200.0, help="Target Z position")
    parser.add_argument("--step_size", type=float, default=-5.0, help="Step size for RK4")
    parser.add_argument("--tag", default="distributions", help="Filename for the plot")
    args = parser.parse_args()

    # ================================================
    # 1. LOAD MAGNETIC FIELD & Get Derivatives for RK4
    # ================================================


    
    f = ROOT.TFile.Open("sim_muon_50GeV_rec.root")
    tree = f.Get("ship_reco_sim")

    # 3. Loop over the events in the tree
    for event_index in range(tree.GetEntries()):
        tree.GetEntry(event_index)
        
        # 4. Loop over the tracks in the current event
        for i, track in enumerate(tree.FitTracks): 
            
            # --- NEW: Filter for Primary Muons ---
            mc_id = tree.fitTrack2MC[i] # Get matched MC truth ID [cite: 594]
            if mc_id >= 0:
                mc_track = tree.MCTrack[mc_id] # [cite: 597]
                # A primary particle from the gun usually has a mother ID of -1
                # PDG 13 or -13 is a muon
                if mc_track.GetMotherId() != -1 or abs(mc_track.GetPdgCode()) != 13:
                    continue # Skip secondary particles or non-muons
            else:
                continue # Skip fake tracks/ghosts
            
            status = track.getFitStatus() 
            if not status.isFitConverged(): 
                continue
                
            ndf = status.getNdf()
            target_z = args.UBT_z 
            
            # Use the tool
            success, ext_pos, ext_mom = TrackExtrapolateTool.extrapolateToPlane(track, target_z)
            
            if success:
                print(f"Event {event_index}: SUCCESS! Z={target_z}: X={ext_pos.X():.2f}, Y={ext_pos.Y():.2f}")
            else:
                print(f"Event {event_index}: Failed. NDF = {ndf} (Tool requires > 20).")
                # If ext_pos is NOT None, it means the RK4 crashed but the linear fallback worked!
                if ext_pos:
                    print(f"  -> Linear fallback position: Z={ext_pos.Z():.2f}, X={ext_pos.X():.2f}, Y={ext_pos.Y():.2f}")

    f.Close()

if __name__ == "__main__":
    main()