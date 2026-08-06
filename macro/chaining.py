import ROOT
import os
import glob

def creating_chains(file_directory, tree_name):
    # Retrieving the simulations
    cuda_muons_sim_dir = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations"
    input_dir = os.path.join(cuda_muons_sim_dir, file_directory)
    file_list = glob.glob(os.path.join(input_dir,"filtered_*.root"))
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
    return output_file

def analysis(file, tree, branch, weighted):
    if not file or not os.path.exists(file):
        print(f"CRITICAL ERROR: Cannot run analysis, file not found at: {file}")
        return

    # Creating paths
    dir_name = os.path.dirname(file)
    out_dir = os.path.join(dir_name,"histograms")
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
            try:
                x = hit.GetX()
                y = hit.GetY()
                z = hit.GetZ()
                pdg = hit.PdgCode()
                track_id = hit.GetTrackID()

            except AttributeError:
                x = hit.fX
                y = hit.fY
                z = hit.fZ
                pdg = hit.fPdgCode
                track_id = hit.fTrackID
                
            # Filling each histogram separately
            abs_pdg = abs(pdg)
            w = 1.0 
            if weighted:
                if track_id >= 0 and track_id < len(ttree.MCTrack):
                    mc_particle = ttree.MCTrack[track_id]
                    w = mc_particle.GetWeight()

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

    # Drawing and Saving the Histograms
    c = ROOT.TCanvas("c", "Canvas", 800, 600)
    suffix = "_weighted.pdf" if weighted else ".pdf"

    h_muons.Draw("COLZ")
    c.SaveAs(os.path.join(out_dir, f"muons_hits{suffix}"))
    
    h_electrons.Draw("COLZ")
    c.SaveAs(os.path.join(out_dir, f"electrons_hits{suffix}"))
    
    h_photons.Draw("COLZ")
    c.SaveAs(os.path.join(out_dir, f"photons_hits{suffix}"))
    
    # Safely close the file to prevent memory leaks
    tfile.Close()
    print(f"Histograms successfully generated and saved in: {out_dir}")



def main():
    # Creating the Chains files
    print("=== Merging Filtered Files ===")
    master_filtered = creating_chains("filtered_files", "cbmsim")
    
    print("=== Merging Smeared Files ===")
    master_smeared = creating_chains("smeared", "ship_reco_sim")

    # Creating the histograms
    print("=== Running Analysis Filtered unweighted===")
    if master_filtered:
        analysis(master_filtered, "cbmsim", "UpstreamTaggerPoint", False)

    print("=== Running Analysis weighted===")
    if master_filtered:
        analysis(master_filtered, "cbmsim", "UpstreamTaggerPoint", True)

if __name__ == "__main__":

    main()