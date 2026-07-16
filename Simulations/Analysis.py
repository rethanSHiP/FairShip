import argparse
import os
import glob
import uproot
import ROOT
import numpy as np
import pandas as pd
import re


def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--sim_file", default="energy_scan/sim_*.root", help="Path to simulation files")
    parser.add_argument("--tag", default="distributions", help="Filename for the plot")
    args = parser.parse_args()

    # ================================================
    # 1. Primary Plots
    # ================================================
    # Styling the Plots------------------------------- 
    output_dir = "histograms"
    os.makedirs(output_dir, exist_ok=True)
    ROOT.gROOT.SetBatch(True)

    # Loading and sorting the data--------------------
    def extract_energy_from_filename(filepath):
        match = re.search(r'_(\d+)GeV\.root', filepath)
        if match:
            return float(match.group(1))
        return 0.0

    simulation_paths = glob.glob("processed_data/B_sim_muon_*.root")
    simulation_paths.sort(key=extract_energy_from_filename)
    pdf_name_hits = os.path.join(output_dir, "UBT_Distributions.pdf")
    pdf_name_momentum = os.path.join(output_dir, "UBT_Momentum.pdf")


    # Create the histogram to store the statistics----
    graphs = {
        "UBT_X_RMS": ROOT.TGraphErrors(),
        "UBT_Y_RMS": ROOT.TGraphErrors(),
        "UBT_X_SIG": ROOT.TGraphErrors(),
        "UBT_Y_SIG": ROOT.TGraphErrors(),
    }

    graphs_mean = {"UBT_X_MEAN": ROOT.TGraphErrors(),
                   "UBT_Y_MEAN": ROOT.TGraphErrors(),}

    graphs_theta= {"theta_X_RMS": ROOT.TGraphErrors(),
                   "theta_Y_RMS": ROOT.TGraphErrors(),
                   "theta_X_SIG": ROOT.TGraphErrors(),
                   "theta_Y_SIG": ROOT.TGraphErrors(),}

    graphs_mean_theta = {"theta_X_MEAN": ROOT.TGraphErrors(),
                         "theta_Y_MEAN": ROOT.TGraphErrors(),}

    # Stiling the histograms----------------------
    ROOT.gStyle.SetOptTitle(1) # Keep title, but we can customize
    ROOT.gStyle.SetLabelSize(0.04, "XY") # Larger axis labels
    ROOT.gStyle.SetTitleSize(0.05, "XY") # Larger axis titles
    ROOT.gStyle.SetOptFit(1111)

    ROOT.gStyle.SetPadLeftMargin(0.15)  # Gives more room on the left for the Y-axis title
    ROOT.gStyle.SetPadRightMargin(0.15) # Keeps the right side tight

    # Retrieve the actual data--------------------

    def bin_width(data,r):
        "Using the Freedman-Diaconis rule for dynamically selecting the bin size"
        #Compute the IQR
        q_75, q_25 = np.percentile(data,[75,25])
        iqr = q_75 - q_25

        N = len(data)

        fd_width = 2 * iqr / (N**(1/3)) if iqr > 0 else 0.0

        total_range = np.max(data) - np.min(data)
        min_allowed_width = total_range / 100.0 if total_range > 0 else 0.01

        return round(max(fd_width, min_allowed_width),r)

    def proper_bins(data, n_min = 10, n_max=500, r=3):
        bin_size = bin_width(data,r)
        sigma, mu = np.std(data), np.mean(data)
        min_data, max_data = mu - 3*sigma, mu + 3*sigma
        n_bins_data = int(np.ceil(np.abs(max_data-min_data)/bin_size))
        n_bins_data = max(n_min, min(n_bins_data, n_max))

        return n_bins_data, min_data, max_data, bin_size

    def tan_rel(pi_reco, pz_reco,pi_true, pz_true):
        tan_true = pi_true/pz_true
        tan_reco = pi_reco/pz_reco
        tan_rel = (tan_reco - tan_true)/(1 + tan_reco*tan_true)

        return tan_rel


    c0 = ROOT.TCanvas("c0","",1200,800)
    c0.Print(f"{pdf_name_hits}[")

    c01 = ROOT.TCanvas("c01","",1200,800)
    c01.Print(f"{pdf_name_momentum}[")

    # Process each simulation file
    for i,sim in enumerate(simulation_paths): 
        print(f"Processing simulation: {sim})")

        # Open the file to retrieve the data 
        with uproot.open(sim) as f:
            if "UBT_Muons" not in f:
                print(f"  -> Skipping: 'UBT_Muons' tree not found.")
                continue
           
            df_UBT = f["UBT_Muons"].arrays(["X", "Y", "Z","X_true","Y_true","Z_true","PX", "PY", "PZ","PX_true","PY_true", "PZ_true"], library="np")
            if len(df_UBT) < 2: continue 

            x_rel = (df_UBT["X"]-df_UBT["X_true"])*10 # mm
            y_rel = (df_UBT["Y"]-df_UBT["Y_true"])*10 # mm
            z_rel = (df_UBT["Z"]-df_UBT["Z_true"])*10 # mm

            px, py, pz = df_UBT["PX"], df_UBT["PY"], df_UBT["PZ"]
            momentum = np.sqrt(px**2+py**2+pz**2)
            momentum_avg = np.mean(np.array(momentum))

            tg_theta_x = tan_rel(px,pz,df_UBT["PX_true"],df_UBT["PZ_true"]) 
            tg_theta_y = tan_rel(py,pz,df_UBT["PY_true"],df_UBT["PZ_true"])

            #Plotting range
            n_bins_x, min_x, max_x, bin_size_x = proper_bins(x_rel, r=3)
            n_bins_y, min_y, max_y, bin_size_y = proper_bins(y_rel, r=3)

            n_bins_theta_x, min_theta_x, max_theta_x, bin_size_theta_x = proper_bins(tg_theta_x, r =9)
            n_bins_theta_y, min_theta_y, max_theta_y, bin_size_theta_y = proper_bins(tg_theta_y, r =9)

            #Creating the histogram
            h_x_ubt = ROOT.TH1D(f"h_x_ubt{i}", "", n_bins_x, min_x, max_x )
            h_y_ubt = ROOT.TH1D(f"h_y_ubt{i}", "", n_bins_y, min_y, max_y )

            h_theta_x = ROOT.TH1D(f"h_theta_x{i}", "", n_bins_theta_x, min_theta_x, max_theta_x)
            h_theta_y = ROOT.TH1D(f"h_theta_y{i}", "", n_bins_theta_y, min_theta_y, max_theta_y)

            #Filling the histogram
            weights = np.ones(len(x_rel), dtype=np.float64)
            h_x_ubt.FillN(len(x_rel), np.array(x_rel, dtype =np.float64), weights)
            h_y_ubt.FillN(len(y_rel), np.array(y_rel, dtype =np.float64), weights)
            h_theta_x.FillN(len(tg_theta_x), np.array(tg_theta_x, dtype =np.float64), weights)
            h_theta_y.FillN(len(tg_theta_y), np.array(tg_theta_y, dtype =np.float64), weights)

            # Fitting the histograms with a Gaussian distribution
            h_x_ubt.Fit("gaus","Q") 
            h_y_ubt.Fit("gaus","Q")
            h_theta_x.Fit("gaus","Q")
            h_theta_y.Fit("gaus","Q")
            
            # Retrieving the statistics for the data
            for name in ["UBT_X", "UBT_Y"]:
                idx = graphs[f"{name}_RMS"].GetN()
            
                # Fill RMS
                graphs[f"{name}_RMS"].SetPoint(idx, momentum_avg, h_x_ubt.GetRMS() if "X" in name else h_y_ubt.GetRMS())
                graphs[f"{name}_RMS"].SetPointError(idx, 0, h_x_ubt.GetRMSError() if "X" in name else h_y_ubt.GetRMSError())
                
                # Fill Sigma
                fit = h_x_ubt.GetFunction("gaus") if "X" in name else h_y_ubt.GetFunction("gaus")
                graphs[f"{name}_SIG"].SetPoint(idx, momentum_avg, fit.GetParameter(2))
                graphs[f"{name}_SIG"].SetPointError(idx, 0, fit.GetParError(2))

                # Fill Mean
                graphs_mean[f"{name}_MEAN"].SetPoint(idx, momentum_avg, h_x_ubt.GetMean() if "X" in name else h_y_ubt.GetMean())
                graphs_mean[f"{name}_MEAN"].SetPointError(idx, 0, h_x_ubt.GetMeanError() if "X" in name else h_y_ubt.GetMeanError())

            for name in ["theta_X", "theta_Y"]:
                idx = graphs_theta  [f"{name}_RMS"].GetN()
            
                # Fill RMS
                graphs_theta[f"{name}_RMS"].SetPoint(idx, momentum_avg, h_theta_x.GetRMS() if "X" in name else h_theta_y.GetRMS())
                graphs_theta[f"{name}_RMS"].SetPointError(idx, 0, h_theta_x.GetRMSError() if "X" in name else h_theta_y.GetRMSError())
                
                # Fill Sigma
                fit = h_theta_x.GetFunction("gaus") if "X" in name else h_theta_y.GetFunction("gaus")
                graphs_theta[f"{name}_SIG"].SetPoint(idx, momentum_avg, fit.GetParameter(2))
                graphs_theta[f"{name}_SIG"].SetPointError(idx, 0, fit.GetParError(2))

                # Fill Mean
                graphs_mean_theta[f"{name}_MEAN"].SetPoint(idx, momentum_avg, h_theta_x.GetMean() if "X" in name else h_theta_y.GetMean())
                graphs_mean_theta[f"{name}_MEAN"].SetPointError(idx, 0, h_theta_x.GetMeanError() if "X" in name else h_theta_y.GetMeanError())


            #Plotting X and Y Hits at UBT
            c0.Clear()
            c0.Divide(2,1)

            c0.cd(1); h_x_ubt.SetTitle(f"UBT X Distribution - {os.path.basename(sim)};X [mm]; counts [{bin_size_x} mm/bin]"); h_x_ubt.Draw()
            c0.cd(2); h_y_ubt.SetTitle(f"UBT Y Distribution - {os.path.basename(sim)};Y [mm]; counts [{bin_size_y} mm/bin]"); h_y_ubt.Draw()
            c0.Print(pdf_name_hits) 
            
            #Plotting px, py, pz at UBT
            c01.Clear()
            c01.Divide(2,1)

            c01.cd(1); h_theta_x.SetTitle(fr"tan(#theta_{{x}}) distribution - {os.path.basename(sim)};tan(#theta_{{x}}); counts [{bin_size_theta_x}/bin]"); h_theta_x.Draw()
            c01.cd(2); h_theta_y.SetTitle(fr"tan(#theta_{{y}}) distribution - {os.path.basename(sim)};tan(#theta_{{y}}); counts [{bin_size_theta_y}/bin]"); h_theta_y.Draw()

            c01.Print(pdf_name_momentum) 
            
    
    c0.Print(f"{pdf_name_hits}]")
    c01.Print(f"{pdf_name_momentum}]")
    print("Data extraction complete!")

    # ==========================================
    # 5. PLOTTING (Overlaying RMS and Sigma)
    # ==========================================
    def style_graph(gr, color, style, marker):
        gr.SetLineColor(color)
        gr.SetMarkerColor(color)
        gr.SetMarkerStyle(marker)
        gr.SetMarkerSize(1.2)
        gr.SetLineWidth(2)
        gr.SetLineStyle(style) 

    # Canvas Setup
    for axis in ["UBT_X", "UBT_Y"]:
        c = ROOT.TCanvas(f"c_{axis}", "", 800, 600)
        c.SetGrid()
        
        gr_rms = graphs[f"{axis}_RMS"]
        gr_sig = graphs[f"{axis}_SIG"]
        
        # Style
        style_graph(gr_rms, ROOT.kBlue+1, 1, 20)
        style_graph(gr_sig, ROOT.kRed+1, 1, 21)
        
        # Setup Fits
        fit_rms = ROOT.TF1(f"fit_rms_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_rms.SetParameters(0, 100, 1)
        fit_rms.SetLineColor(ROOT.kBlue+1)
        
        fit_sig = ROOT.TF1(f"fit_sig_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_sig.SetParameters(0, 100, 1)
        fit_sig.SetLineColor(ROOT.kRed+1)
        
        # Fit with "S" (Store result) and draw
        gr_rms.Fit(fit_rms, "Q R S")
        gr_sig.Fit(fit_sig, "Q R S")
        
        gr_rms.SetTitle(f"{axis} Spread Comparison;momentum [GeV];Spread [mm]")
        gr_rms.Draw("AP")
        c.Update()

        stats_rms = gr_rms.FindObject("stats")
        if stats_rms:
            stats_rms.SetX1NDC(0.65); stats_rms.SetX2NDC(0.90)
            stats_rms.SetY1NDC(0.60); stats_rms.SetY2NDC(0.85)
            stats_rms.SetTextColor(ROOT.kBlue+1)
            stats_rms.SetLineColor(ROOT.kBlue+1)

        gr_sig.Draw("P SAMES")
        c.Update()

        stats_sig = gr_sig.FindObject("stats")
        if stats_sig:
            stats_sig.SetX1NDC(0.65); stats_sig.SetX2NDC(0.90)
            stats_sig.SetY1NDC(0.30); stats_sig.SetY2NDC(0.55)
            stats_sig.SetTextColor(ROOT.kRed+1)
            stats_sig.SetLineColor(ROOT.kRed+1)

        c.Update()

        # Legend
        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_rms, "RMS", "lep")
        leg.AddEntry(gr_sig, "Gaussian Sigma", "lep")
        leg.SetBorderSize(0)
        leg.Draw()
        
        c.Modified()
        c.Update()
        c.SaveAs(os.path.join(output_dir, f"{axis}_STATS.png"))

    #--------- MEAN----------
    for axis in ["UBT_X", "UBT_Y"]:
        c2 = ROOT.TCanvas(f"c2_{axis}", "", 900, 700)
        c2.SetGrid()
        
        gr_mean = graphs_mean[f"{axis}_MEAN"]
        
        # Style
        style_graph(gr_mean, ROOT.kOrange+1, 1, 20)
        
        gr_mean.SetTitle(f"{axis} Mean Spread;momentum [GeV];Spread [mm]")
        gr_mean.Draw("AP")
        c2.Update()

        stats_mean = gr_mean.FindObject("stats")
        if stats_mean:
            stats_mean.SetX1NDC(0.65); stats_mean.SetX2NDC(0.90)
            stats_mean.SetY1NDC(0.60); stats_mean.SetY2NDC(0.85)
            stats_mean.SetTextColor(ROOT.kOrange+1)
            stats_mean.SetLineColor(ROOT.kOrange+1)
        # Legend
        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_mean, "Mean", "lep")
        leg.SetBorderSize(0)
        leg.Draw()
        
        c2.Modified()
        c2.Update()
        c2.SaveAs(os.path.join(output_dir, f"{axis}_MEAN.png"))

    # Angular direction plots 
    # Canvas Setup
    for axis in ["theta_X", "theta_Y"]:
        c = ROOT.TCanvas(f"c_{axis}", "", 800, 600)
        c.SetGrid()
        
        gr_rms = graphs_theta[f"{axis}_RMS"]
        gr_sig = graphs_theta[f"{axis}_SIG"]
        
        # Style
        style_graph(gr_rms, ROOT.kBlue+1, 1, 20)
        style_graph(gr_sig, ROOT.kRed+1, 1, 21)
        
        # Setup Fits
        fit_rms = ROOT.TF1(f"fit_rms_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_rms.SetParameters(0, 100, 1)
        fit_rms.SetLineColor(ROOT.kBlue+1)
        
        fit_sig = ROOT.TF1(f"fit_sig_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_sig.SetParameters(0, 100, 1)
        fit_sig.SetLineColor(ROOT.kRed+1)
        
        # Fit with "S" (Store result) and draw
        gr_rms.Fit(fit_rms, "Q R S")
        gr_sig.Fit(fit_sig, "Q R S")
        
        gr_rms.SetTitle(fr"{axis} Direction comparison ;momentum [GeV];tan(#theta_{axis[-1:]})")
        gr_rms.Draw("AP")
        c.Update()

        stats_rms = gr_rms.FindObject("stats")
        if stats_rms:
            stats_rms.SetX1NDC(0.65); stats_rms.SetX2NDC(0.90)
            stats_rms.SetY1NDC(0.60); stats_rms.SetY2NDC(0.85)
            stats_rms.SetTextColor(ROOT.kBlue+1)
            stats_rms.SetLineColor(ROOT.kBlue+1)

        gr_sig.Draw("P SAMES")
        c.Update()

        stats_sig = gr_sig.FindObject("stats")
        if stats_sig:
            stats_sig.SetX1NDC(0.65); stats_sig.SetX2NDC(0.90)
            stats_sig.SetY1NDC(0.30); stats_sig.SetY2NDC(0.55)
            stats_sig.SetTextColor(ROOT.kRed+1)
            stats_sig.SetLineColor(ROOT.kRed+1)

        c.Update()

        # Legend
        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_rms, "RMS", "lep")
        leg.AddEntry(gr_sig, "Gaussian Sigma", "lep")
        leg.SetBorderSize(0)
        leg.Draw()
        
        c.Modified()
        c.Update()
        c.SaveAs(os.path.join(output_dir, f"{axis}_STATS.png"))

    #--------- MEAN----------
    for axis in ["theta_X", "theta_Y"]:
        c2 = ROOT.TCanvas(f"c2_{axis}", "", 900, 700)
        c2.SetGrid()
        
        gr_mean = graphs_mean_theta[f"{axis}_MEAN"]
        
        # Style
        style_graph(gr_mean, ROOT.kOrange+1, 1, 20)
        
        gr_mean.SetTitle(fr"{axis} Mean Direction ;momentum [GeV];tan(#theta_{axis[-1:]})")
        gr_mean.Draw("AP")
        c2.Update()

        stats_mean = gr_mean.FindObject("stats")
        if stats_mean:
            stats_mean.SetX1NDC(0.65); stats_mean.SetX2NDC(0.90)
            stats_mean.SetY1NDC(0.60); stats_mean.SetY2NDC(0.85)
            stats_mean.SetTextColor(ROOT.kOrange+1)
            stats_mean.SetLineColor(ROOT.kOrange+1)
        # Legend
        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_mean, "Mean", "lep")
        leg.SetBorderSize(0)
        leg.Draw()
        
        c2.Modified()
        c2.Update()
        c2.SaveAs(os.path.join(output_dir, f"{axis}_MEAN.png"))

        print(180/np.pi * np.arctan(2/50))
if __name__ == "__main__":
    main()