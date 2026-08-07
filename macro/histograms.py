import ROOT
import os
import glob

def produce_histograms(histo_dir):
    dir_cuda_muons = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations"
    dir_hist = os.path.join(dir_cuda_muons,histo_dir)
    list_hists = glob.glob(os.path.join(dir_hist,"raw_hists_*.root"))
    out_name = os.path.join(dir_hist,"stacked.root")

    if os.path.exists(out_name):
        print(f"Removing old {out_name} to start fresh...")
        os.remove(out_name)

    merger = ROOT.TFileMerger(False)
    merger.OutputFile(out_name, "RECREATE")

    for i, hist_file in enumerate(list_hists):
        if i % 50 == 0:
            print(f"  ... loaded {i} / {len(list_hists)} files")
        merger.AddFile(hist_file)

    # Execute the merge! ROOT automatically finds h_muons, h_electrons, etc., and adds their bins.
    merger.Merge()
    
    print(f"Success! Master histogram file saved to: {out_name}\n")
    return out_name

def analyze_hist(file, weight_label):
    file_dir = os.path.dirname(file)
    out_dir = os.path.join(file_dir,"histograms")

    os.makedirs(out_dir, exist_ok=True)

    f_in = ROOT.TFile.Open(file, "READ")

    # 2. Extract the histograms from the file by their exact names
    h_muons = f_in.Get("h_muons")
    h_electrons = f_in.Get("h_electrons")
    h_photons = f_in.Get("h_photons")

    # 3. Set up the canvas
    c = ROOT.TCanvas("c", "Canvas", 800, 600)

    if h_muons:
        h_muons.Draw("COLZ")
        nx = h_muons.GetNbinsX()
        ny = h_muons.GetNbinsY()
        total_muons_nw = h_muons.Integral(0, nx + 1, 0, ny + 1)
        print(f"In this simulation there are: {total_muons_nw} muons in total")
        c.SaveAs(os.path.join(out_dir, f"muons_distribution_{weight_label}.pdf"))
        
    if h_electrons:
        h_electrons.Draw("COLZ")
        c.SaveAs(os.path.join(out_dir, f"electrons_distribution_{weight_label}.pdf"))
        
    if h_photons:
        h_photons.Draw("COLZ")
        c.SaveAs(os.path.join(out_dir, f"photons_distribution_{weight_label}.pdf"))

    f_in.Close()
    print(f"Finished drawing {weight_label} plots! Saved in: {out_dir}\n")

def main():

    unweighted = produce_histograms("raw_histograms_unweighted")
    weighted = produce_histograms("raw_histograms_weighted")

    analyze_hist(unweighted,"unweighted")
    analyze_hist(weighted,"weighted")

if __name__ == "__main__":
    main()