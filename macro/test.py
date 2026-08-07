import os
import sys
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True) # Don't try to open X11 graphics window

def run_mirror_test():
    det_file = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/extrapolated/extrapolated.root" # or extrapolated_with_noise.root
    truth_file = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/filtered_files/chained_filtered_files.root"

    print("Opening files for the Mirror Test...")
    f_det = ROOT.TFile.Open(det_file, "READ")
    f_truth = ROOT.TFile.Open(truth_file, "READ")

    t_det = f_det.Get("ship_reco_sim")
    t_truth = f_truth.Get("cbmsim")

    # Create the histograms
    # X bounds set from -250 to 250 cm (the width of the UBT)
    h_truth = ROOT.TH1D("h_truth", "Muon (#mu^{-}) X-Position at UBT; X [cm]; Events", 100, -250, 250)
    h_extrap = ROOT.TH1D("h_extrap", "Muon (#mu^{-}) X-Position at UBT; X [cm]; Events", 100, -250, 250)

    n_events = t_truth.GetEntries()
    print(f"Scanning {n_events} events...")

    for i in range(n_events):
        t_det.GetEntry(i)
        t_truth.GetEntry(i)

        # 1. Skip if RK4 extrapolation failed (Check for our -999.0 placeholder!)
        if t_det.extrap_ubt_x == -999.0:
            continue

        # 2. STRICT FILTER: Only accept pure Muons (PDG == 13)
        primary_particle = t_truth.MCTrack[0]
        try:
            pdg = primary_particle.GetPdgCode()
        except AttributeError:
            pdg = primary_particle.fPdgCode
            
        if pdg != -13:
            continue # Skip Anti-Muons (-13) and background particles

        # 3. Find the valid True Hit at the final UBT plane
        valid_truth_hit = None
        max_z = -9999.0
        
        for hit in t_truth.UpstreamTaggerPoint:
            try:
                track_id = hit.GetTrackID()
                hit_z = hit.GetZ()
            except AttributeError:
                track_id = hit.fTrackID
                hit_z = hit.fZ
                
            if track_id == 0 and hit_z > max_z:
                max_z = hit_z
                valid_truth_hit = hit

        if valid_truth_hit and max_z > 3265.0:
            # 4. Extract Truth X safely
            try:
                truth_x = valid_truth_hit.GetX()
            except AttributeError:
                truth_x = valid_truth_hit.fX
                
            # Fill the histograms!
            h_truth.Fill(truth_x)
            h_extrap.Fill(t_det.extrap_ubt_x)

    print("Drawing canvas...")
    c1 = ROOT.TCanvas("c1", "Mirror Test", 800, 600)
    c1.SetGrid()

    # Style the Truth (Green, Filled)
    h_truth.SetLineColor(ROOT.kGreen + 2)
    h_truth.SetFillColorAlpha(ROOT.kGreen + 2, 0.5)
    h_truth.SetLineWidth(2)

    # Style the Extrapolated (Red, Thick Line)
    h_extrap.SetLineColor(ROOT.kRed)
    h_extrap.SetLineWidth(3)

    # Draw them together
    # Check which one is taller so the Y-axis scales correctly
    max_y = max(h_truth.GetMaximum(), h_extrap.GetMaximum())
    h_truth.SetMaximum(max_y * 1.2)

    h_truth.Draw("HIST")
    h_extrap.Draw("HIST SAME")

    # Add a legend
    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.AddEntry(h_truth, "True GEANT4 #mu^{-}", "f")
    leg.AddEntry(h_extrap, "RK4 Extrapolated #mu^{-}", "l")
    leg.SetBorderSize(0)
    leg.Draw()

    # Save
    out_img = "mirror_test_muons.png"
    c1.SaveAs(out_img)
    print(f"Done! Check the image: {out_img}")


import sys
import ROOT

def scan_geometry(geo_file_path):
    print(f"Scanning geometry file: {geo_file_path}")
    f = ROOT.TFile.Open(geo_file_path, "READ")
    
    if not f or f.IsZombie():
        print("Error: Could not open file.")
        sys.exit(1)

    print("\n--- Top Level Objects ---")
    keys = f.GetListOfKeys()
    
    field_found = False
    for key in keys:
        obj_name = key.GetName()
        obj_class = key.GetClassName()
        print(f"Found: {obj_name} (Class: {obj_class})")
        
        # FairShip stores the magnetic field as a 'ShipBFieldMap' or 'ShipConstField'
        if "BField" in obj_class or "Field" in obj_name:
            field_found = True
            print("\n" + "="*50)
            print(f">>> MAGNETIC FIELD OBJECT DETECTED: {obj_name} <<<")
            field_obj = f.Get(obj_name)
            
            # The Print() method usually dumps the file map name and properties
            field_obj.Print() 
            print("="*50 + "\n")
            
    # Sometimes it's hidden inside the FairBaseParSet
    par_set = f.Get("FairBaseParSet")
    if par_set:
        print("\n--- Checking Parameter Set ---")
        par_set.Print()

    if not field_found:
        print("\nNo explicit magnetic field object found in the top directory.")
        print("Try scanning the 'ship.params.root' file instead of the geofile.")

    f.Close()

if __name__ == "__main__":
    # Replace with the path to the cuda_muons geofile or params file
    geo_path = "/eos/experiment/ship/simulation/cuda_muons/try_2025//processed_muons/geo_cuda_test.root"
    #scan_geometry(geo_path)
    run_mirror_test()
