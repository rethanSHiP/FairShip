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

def smearing(filtered_file,out_dir):
    # Same Geoemtry for all cuda muons
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
    out_smear = os.path.join(BASE_OUTPUT_DIR, "smeared")

    os.makedirs(out_filt, exist_ok=True)
    os.makedirs(out_smear, exist_ok=True)
    
    filtered_file = os.path.join(out_filt, args.out_file_name)

    # Filtering and smearing the simulation file
    filter_events(args.input, filtered_file)
    smearing(filtered_file, out_smear)

if __name__ == "__main__":
    main()