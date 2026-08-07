import ROOT
import os
import glob
import subprocess
import shutil

def creating_chains(file_directory, tree_name):
    # Retrieving the simulations
    cuda_muons_sim_dir = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations"
    input_dir = os.path.join(cuda_muons_sim_dir, file_directory)
    file_list = glob.glob(os.path.join(input_dir,"filtered_*.root"))
    file_list.sort(key=lambda f: f.replace('_rec', ''))
    output_file = os.path.join(input_dir,f"chained_{file_directory}.root")

    # Creating the Chain to merge the files
    chain = ROOT.TChain(tree_name)

    if not file_list:
        print("No ROOT files found to merge!")
        return
    print(f"Found {len(file_list)} files. Building TChain...")

    # Add every file to the chain
    for f in file_list:
        chain.Add(f)
    total_events = chain.GetEntries()
    print(f"TChain successfully built with {total_events} total events.")
    print("Merging into a single file... (This might take a minute)")
    
    # 5. Use the built-in Merge command to write the chain to disk
    chain.Merge(output_file)
    
    print(f"Success! Master file saved to: {output_file}")


def main():
    # Creating the Chains files
    print("=== Merging Filtered Files ===")
    creating_chains("filtered_files", "cbmsim")

    print("=== Merging Filtered Files ===")
    creating_chains("smeared", "ship_reco_sim")

if __name__ == "__main__":
    main()