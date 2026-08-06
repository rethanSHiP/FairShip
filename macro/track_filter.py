import ROOT
import os
import sys
import numpy as np
import uproot

try:
    import shipRoot_conf
    import shipunit as uç

except ImportError:
    print("CRITICAL ERROR: FairShip environment not loaded.")
    print("Please run 'alienv enter FairShip/latest' first!")
    sys.exit(1)

def extract_matched_data(): 
    shipRoot_conf.configure()

    path_to_reco =  "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/smeared/chained_smeared.root"
    path_to_truth = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/filtered_files/chained_filtered_files.root"

    # 1. Open both files safely
    f_reco = ROOT.TFile.Open(path_to_reco, "READ")
    f_truth = ROOT.TFile.Open(path_to_truth, "READ")

    if not f_reco or f_reco.IsZombie() or not f_truth or f_truth.IsZombie():
        print("CRITICAL ERROR: Could not open one or both ROOT files.")
        sys.exit(1)

    tree_reco = f_reco.Get("ship_reco_sim")
    tree_truth = f_truth.Get("cbmsim")

    # Arrays to store perfectly matched data
    x_sst, y_sst, z_sst = [], [], []
    px_sst, py_sst, pz_sst = [], [], []

    x_ubt, y_ubt, z_ubt = [], [], []
    px_ubt, py_ubt, pz_ubt = [], [], []

    n_entries = tree_truth.GetEntries()
    print(f"Scanning {n_entries} events for matched SST-UBT data...")

    events_matched = 0

    # 2. The SINGLE Synchronized Event Loop
    for i in range(n_entries):
        tree_reco.GetEntry(i)
        tree_truth.GetEntry(i)

        # Step A: Find the valid Reconstructed Track in the SST
        valid_reco_track = None
        if hasattr(tree_reco, "FitTracks") and len(tree_reco.FitTracks) > 0:
            track = tree_reco.FitTracks[0] # Grab the first track
            if track.getFitStatus().isFitConverged():
                valid_reco_track = track
        
        # Step B: Find the First Truth Hit in the UBT
        valid_truth_hit = None
        for hit in tree_truth.UpstreamTaggerPoint:
            if hit.GetTrackID() == 0:  # Primary muon only
                valid_truth_hit = hit
                break # Stop at the first layer of the UBT

        # Step C: ONLY save the data if BOTH exist in this event!
        if valid_reco_track and valid_truth_hit:
            
            # --- Extract SST Reco Data ---
            state = valid_reco_track.getFittedState()
            pos, mom = state.getPos(), state.getMom()
            x_sst.append(pos.x())
            y_sst.append(pos.y())
            z_sst.append(pos.z())

            px_sst.append(mom.x())
            py_sst.append(mom.y())
            pz_sst.append(mom.z())

            # --- Extract UBT Truth Data ---
            x_ubt.append(valid_truth_hit.LastPoint().x())
            y_ubt.append(valid_truth_hit.LastPoint().y())
            z_ubt.append(valid_truth_hit.LastPoint().z())

            px_ubt.append(valid_truth_hit.LastPoint().Px())
            py_ubt.append(valid_truth_hit.LastPoint().Py())
            pz_ubt.append(valid_truth_hit.LastPoint().Pz())  

            events_matched += 1
            
        # Clean up memory pointers (good practice in PyROOT)
        if hasattr(tree_reco, "FitTracks") and hasattr(tree_reco.FitTracks, "clear"):
            tree_reco.FitTracks.clear()

    print(f"Extraction complete! Successfully matched {events_matched} out of {n_entries} events.")

    f_reco.Close()
    f_truth.Close()

    # Convert to numpy arrays for easy Runge-Kutta math later
    sst_data = {
        "pos": [np.array(x_sst), np.array(y_sst), np.array(z_sst)],
        "mom": [np.array(px_sst), np.array(py_sst), np.array(pz_sst)]
    }
    
    ubt_data = {
        "pos": [np.array(x_ubt), np.array(y_ubt), np.array(z_ubt)],
        "mom": [np.array(px_ubt), np.array(py_ubt), np.array(pz_ubt)]
    }

    return sst_data, ubt_data

def main():
    sst_data, ubt_data = extract_matched_data()
    directory = "cuda_muons_simulations/matched"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, "truth_and_detector.root")

    # Uproot can write dictionaries of NumPy arrays directly to a TTree in one line!
    with uproot.recreate(output_file) as f_out:
        f_out["cbmsim"] = {
            "sst_x": sst_data["pos"][0],
            "sst_y": sst_data["pos"][1],
            "sst_z": sst_data["pos"][2],
            "sst_px": sst_data["mom"][0],
            "sst_py": sst_data["mom"][1],
            "sst_pz": sst_data["mom"][2],
            
            "ubt_x": ubt_data["pos"][0],
            "ubt_y": ubt_data["pos"][1],
            "ubt_z": ubt_data["pos"][2],
            "ubt_px": ubt_data["mom"][0],
            "ubt_py": ubt_data["mom"][1],
            "ubt_pz": ubt_data["mom"][2]
        }

    print(f"Successfully saved matched data to {output_file}")

if __name__ == "__main__":
    main()
