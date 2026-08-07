import ROOT
import os
import sys
import subprocess
import shutil
from argparse import ArgumentParser

def filter_events(input_file, filtered_out_file):
    #############################################
    ############ 1. CREATING FILES ##############
    #############################################
    print(f"--- FILTERING: {input_file} ---")

    # Opening the ROOT file for cuda_muons
    f_in = ROOT.TFile.Open(input_file, "READ") 
    if not f_in or f_in.IsZombie():
        print("CRITICAL ERROR: Input ROOT file is corrupted or unreadable.")
        sys.exit(1)

    # Opening the corresponding Tree 
    tree_in = f_in.Get("cbmsim")
    if not tree_in:
        print("CRITICAL ERROR: TTree 'cbmsim' not found in the input file.")
        f_in.Close()
        sys.exit(1)

    # Creating output filtered ROOT file
    f_out = ROOT.TFile.Open(filtered_out_file, "RECREATE")
    tree_out = tree_in.CloneTree(0) 

    print("Scanning geometry to find the final UBT surface...")
    max_z = -9999.0
    for i in range(min(100, tree_in.GetEntries())):
        tree_in.GetEntry(i)
        for hit in tree_in.UpstreamTaggerPoint:
            if hit.GetZ() > max_z:
                max_z = hit.GetZ()

    last_ubt_z = round(max_z, 1)
    print(f"Detected last UBT surface at exactly Z ≈ {last_ubt_z} cm")

    #############################################
    ############ 2. FILTERING THE DATA ##########
    #############################################

    try:
        # Obtaining the total number of events
        n_entries = tree_in.GetEntries()

        # Looping over all events
        for i in range(n_entries):
            tree_in.GetEntry(i)
            
            has_primary_sst_hit = False
            has_primary_ubt_hit = False

            # Looping over all the hits at each event
            for hit in tree_in.strawtubesPoint:
                if hit.GetTrackID() == 0:
                    has_primary_sst_hit = True
                    break

            for hit in tree_in.UpstreamTaggerPoint:
                if hit.GetTrackID() == 0:
                    if hit.GetZ() >= (last_ubt_z - 1.0):
                        has_primary_ubt_hit = True
                        break
            
            # Keeping the interesting events
            if has_primary_sst_hit and has_primary_ubt_hit:
                tree_out.Fill()
                
        print(f"Events kept (primary muon reached both): {tree_out.GetEntries()} out of {n_entries} total events.")
        tree_out.Write()

    # Closing the files
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
    finally:
        f_out.Close()
        f_in.Close()
        print("Files safely closed.")

def analysis(file, tree, branch, weighted, out_dir, out_filename):
    if not file or not os.path.exists(file):
        print(f"CRITICAL ERROR: Cannot run analysis, file not found at: {file}")
        return

    # Creating paths
    os.makedirs(out_dir, exist_ok=True)

    # Opening the file and Tree
    tfile = ROOT.TFile.Open(file,"READ")
    ttree = tfile.Get(tree)

    # Creating empty histograms and filters
    h_muons =     ROOT.TH2D("h_muons",rf"Muons hits (per cm^2]); X [cm]; Y [cm]", 440, -220, 220, 640, -320,320)
    h_electrons = ROOT.TH2D("h_electrons",rf"Muons electrons (per cm^2]); X [cm]; Y [cm]", 500, -220, 220, 100, -320,320)
    h_photons =   ROOT.TH2D("h_photons",rf"Muons photons (per cm^2]); X [cm]; Y [cm]", 440, -220, 220, 640, -320,320)

    w = 1

    n_entries = ttree.GetEntries()
    for i in range(n_entries):
        ttree.GetEntry(i)
        hits = getattr(ttree, branch)

        event_final_hits = {}
        
        for hit in hits:
            # Safely extract coordinates and tracking data matching your cbmsim layout
            x = hit.GetX()
            y = hit.GetY()
            z = hit.GetZ()
            track_id = hit.GetTrackID()
            w = 1.0

            if 0 <= track_id < len(ttree.MCTrack):
                mc_particle = ttree.MCTrack[track_id]
                pdg = mc_particle.GetPdgCode()
                if weighted:
                    w = mc_particle.GetWeight()
                    
            abs_pdg = abs(pdg)

            # Selecting only the last hit
            if track_id not in event_final_hits or z > event_final_hits[track_id]['z']:
                event_final_hits[track_id] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'pdg': abs_pdg,
                    'w': w
                }

        # Filling the histograms
        for hit_data in event_final_hits.values():
            if hit_data['pdg'] == 13:
                h_muons.Fill(hit_data['x'], hit_data['y'], hit_data['w'])
            elif hit_data['pdg'] == 11:
                h_electrons.Fill(hit_data['x'], hit_data['y'], hit_data['w'])
            elif hit_data['pdg'] == 22:
                h_photons.Fill(hit_data['x'], hit_data['y'], hit_data['w'])

    root_out_path = os.path.join(out_dir, out_filename)
    f_out = ROOT.TFile.Open(root_out_path, "RECREATE")
    
    # Writing the histograms saves the bins so they can be stacked later!
    h_muons.Write()
    h_electrons.Write()
    h_photons.Write()
    
    f_out.Close()
    tfile.Close()
    print(f"Histograms successfully generated and saved to: {root_out_path}")


def smearing(filtered_file,out_dir):
    geo_file = "/eos/experiment/ship/simulation/cuda_muons/try_2025/processed_muons/geo_cuda_test.root"
    ship_reco_path = "/afs/cern.ch/work/r/rethan/public/FairShip/macro/ShipReco.py"

    print(f"\n--- Smearing: {filtered_file} ---")

    # Smearing script command
    command = [
        "python", 
        ship_reco_path, 
        "-f", filtered_file, 
        "-g", geo_file
    ]

    # Smearing the filtered files
    try:
        subprocess.run(command, check=True)
        print("Smearing completed successfully!")

        # Moving the file to the desired destination
        base_name = os.path.basename(filtered_file)
        expected_output = base_name.replace(".root", "_rec.root")
        
        if os.path.exists(expected_output): 
            os.makedirs(out_dir, exist_ok=True) # Ensures it exists
            destination = os.path.join(out_dir, expected_output)
                
            shutil.move(expected_output, destination)
            print(f"Success! Smeared file moved to: {destination}")
            
    # Error handling
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: The smearing script crashed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("CRITICAL ERROR: Could not find 'macro/ShipReco.py'. Ensure you run this script from the main FairShip directory.")
        sys.exit(1)

def main():
    parser = ArgumentParser(description="Filter and Smear Simulation Events.")
    
    # Options to run the script
    parser.add_argument("-i", "--input", required = True, help="Input ROOT file")
    parser.add_argument("-o", "--out_file_name", required = True, help="Output directory")
    args = parser.parse_args()

    # Setting the paths
    BASE_OUTPUT_DIR = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations"
    out_filt = os.path.join(BASE_OUTPUT_DIR, "filtered_files")
    out_hists = os.path.join(BASE_OUTPUT_DIR, "raw_histograms_unweighted")
    out_hists_w = os.path.join(BASE_OUTPUT_DIR, "raw_histograms_weighted")
    out_smeared = os.path.join(BASE_OUTPUT_DIR, "filtered_files")

    filtered_file = os.path.join(out_filt, args.out_file_name)
    hist_filename = args.out_file_name.replace("filtered_sim_", "raw_hists_")
    smeared_file = os.path.join(out_smeared, args.out_file_name)

    os.makedirs(out_filt, exist_ok=True)
    os.makedirs(out_hists, exist_ok=True)
    os.makedirs(out_hists_w, exist_ok=True)
    os.makedirs(out_smeared, exist_ok=True)
    
    # Creating fluxes histograms
    analysis(args.input, "cbmsim", "UpstreamTaggerPoint", False, out_hists, hist_filename)
    analysis(args.input, "cbmsim", "UpstreamTaggerPoint", True, out_hists_w, hist_filename)

    # Creating filtered and smeared files
    filter_events(args.input, filtered_file)
    smearing(filtered_file, out_smeared)

if __name__ == "__main__":
    main()