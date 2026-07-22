import argparse
import os
import glob
import uproot
import ROOT
import numpy as np
import pandas as pd
import re
import MC_scat


def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--sim_dir", default="energy_scan_reco", help="Path to simulation files")
    parser.add_argument("--hist_dir", default= None, help="Directory to save the plots")
    parser.add_argument("--sbt", default= None, help="Directory to save the plots")
    args = parser.parse_args()
    if args.hist_dir is None: args.hist_dir = args.sim_dir

    # ================================================
    # 1. Set Up for Plotting
    # ================================================

    # Creating Directories and extracting data to plot

    output_dir = os.path.join("histograms", args.hist_dir)
    os.makedirs(output_dir, exist_ok=True)

    pdf_name_hits = os.path.join(output_dir, "UBT_Distributions.pdf")
    pdf_name_momentum = os.path.join(output_dir, "UBT_Momentum.pdf")

    def extract_energy_from_filename(filepath):
        match = re.search(r'_(\d+)GeV\.root', filepath)
        if match:
            return float(match.group(1))
        return 0.0

    simulation_paths = glob.glob(os.path.join(args.sim_dir, "B_sim_muon_*.root"))
    simulation_paths.sort(key=extract_energy_from_filename)
    
    # Creating the statistics plots-------------------
    axes = ["UBT_X", "UBT_Y", "theta_X", "theta_Y"]
    graph_types = ["RMS", "SIG", "MEAN"]
    
    graphs = {}
    for axis in axes:
        for g_type in graph_types:
            graphs[f"{axis}_{g_type}"] = ROOT.TGraphErrors()

    # Useful functions for Plotting-------------------
    def bin_width(data, r):
        # Using the Freedman-Diaconis rule
        q_75, q_25 = np.percentile(data, [75, 25])
        iqr = q_75 - q_25
        N = len(data)
        fd_width = 2 * iqr / (N**(1/3)) if iqr > 0 else 0.0
        total_range = np.max(data) - np.min(data)
        min_allowed_width = total_range / 100.0 if total_range > 0 else 0.01
        return round(max(fd_width, min_allowed_width), r)

    def proper_bins(data, n_min=10, n_max=500, r=3):
        bin_size = bin_width(data, r)
        sigma, mu = np.std(data), np.mean(data)
        min_data, max_data = mu - 3*sigma, mu + 3*sigma
        n_bins_data = int(np.ceil(np.abs(max_data - min_data) / bin_size))
        n_bins_data = max(n_min, min(n_bins_data, n_max))
        return n_bins_data, min_data, max_data, bin_size

    def tan_rel(pi_reco, pz_reco, pi_true, pz_true):
        tan_true = pi_true / pz_true
        tan_reco = pi_reco / pz_reco
        tan_rel_val = (tan_reco - tan_true) / (1 + tan_reco * tan_true)
        return np.arctan(tan_rel_val)

    # Defining a global plot style--------------------
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetLabelSize(0.04, "XY") 
    ROOT.gStyle.SetTitleSize(0.05, "XY") 
    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetPadLeftMargin(0.15) 
    ROOT.gStyle.SetPadRightMargin(0.15)

    # ================================================
    # 2. Retrieving the information
    # ================================================
    c_hits = ROOT.TCanvas("c_hits", "", 1200, 800)
    c_hits.Print(f"{pdf_name_hits}[")

    c_dir = ROOT.TCanvas("c_dir", "", 1200, 800)
    c_dir.Print(f"{pdf_name_momentum}[")

    #Storing some usefull information
    p_values = []
    distances = []
    x_he_mean = []
    x_sbt_mean = []
    
    for i, sim in enumerate(simulation_paths): 
        print(f"Processing simulation: {sim}")

        with uproot.open(sim) as f:
            if "UBT_Muons" not in f:
                print("  -> Skipping: 'UBT_Muons' tree not found.")
                continue
           
            df_UBT = f["UBT_Muons"].arrays(["X", "Y", "Z", "X_true", "Y_true", "Z_true",
                                            "PX", "PY", "PZ", "PX_true", "PY_true", "PZ_true",
                                            "x_He","x_SBT","distances"], 
                                           library="np")
            if len(df_UBT) < 2: continue 


            # Calculate relative distances and momentum
            x_rel = (df_UBT["X"] - df_UBT["X_true"]) * 10 # mm
            y_rel = (df_UBT["Y"] - df_UBT["Y_true"]) * 10 # mm

            px, py, pz = df_UBT["PX"], df_UBT["PY"], df_UBT["PZ"]
            momentum = np.sqrt(px**2 + py**2 + pz**2)
            momentum_avg = np.mean(momentum) # GeV

            tg_theta_x = tan_rel(px, pz, df_UBT["PX_true"], df_UBT["PZ_true"]) * 1e3 # mrad
            tg_theta_y = tan_rel(py, pz, df_UBT["PY_true"], df_UBT["PZ_true"]) * 1e3 # mrad

            # Store the true distance traveled and energy of each simulation file
            x_he_mean.append(np.mean(df_UBT["x_He"]))
            x_sbt_mean.append(np.mean(df_UBT["x_SBT"]))
            
            distances.append(np.mean(df_UBT["distances"]))
            p_values.append(momentum_avg)

            # Plotting ranges
            n_bins_x, min_x, max_x, bin_size_x = proper_bins(x_rel, r=3)
            n_bins_y, min_y, max_y, bin_size_y = proper_bins(y_rel, r=3)
            n_bins_theta_x, min_theta_x, max_theta_x, bin_size_theta_x = proper_bins(tg_theta_x, r=9)
            n_bins_theta_y, min_theta_y, max_theta_y, bin_size_theta_y = proper_bins(tg_theta_y, r=9)

            # Creating the histograms
            h_x_ubt = ROOT.TH1D(f"h_x_ubt{i}", "", n_bins_x, min_x, max_x)
            h_y_ubt = ROOT.TH1D(f"h_y_ubt{i}", "", n_bins_y, min_y, max_y)
            h_theta_x = ROOT.TH1D(f"h_theta_x{i}", "", n_bins_theta_x, min_theta_x, max_theta_x)
            h_theta_y = ROOT.TH1D(f"h_theta_y{i}", "", n_bins_theta_y, min_theta_y, max_theta_y)

            # Filling the histograms (Excellent use of FillN!)
            weights = np.ones(len(x_rel), dtype=np.float64)
            h_x_ubt.FillN(len(x_rel), np.array(x_rel, dtype=np.float64), weights)
            h_y_ubt.FillN(len(y_rel), np.array(y_rel, dtype=np.float64), weights)
            h_theta_x.FillN(len(tg_theta_x), np.array(tg_theta_x, dtype=np.float64), weights)
            h_theta_y.FillN(len(tg_theta_y), np.array(tg_theta_y, dtype=np.float64), weights)

            # Fitting
            h_x_ubt.Fit("gaus", "Q") 
            h_y_ubt.Fit("gaus", "Q")
            h_theta_x.Fit("gaus", "Q")
            h_theta_y.Fit("gaus", "Q")
            
            # OPTIMIZED: Retrieving statistics using a mapped dictionary
            hist_map = {
                "UBT_X": h_x_ubt,
                "UBT_Y": h_y_ubt,
                "theta_X": h_theta_x,
                "theta_Y": h_theta_y
            }
            
            for axis, hist in hist_map.items():
                idx = graphs[f"{axis}_RMS"].GetN()
                
                # Fill RMS
                graphs[f"{axis}_RMS"].SetPoint(idx, momentum_avg, hist.GetRMS())
                graphs[f"{axis}_RMS"].SetPointError(idx, 0, hist.GetRMSError())
                
                # Fill Mean
                graphs[f"{axis}_MEAN"].SetPoint(idx, momentum_avg, hist.GetMean())
                graphs[f"{axis}_MEAN"].SetPointError(idx, 0, hist.GetMeanError())
                
                # Fill Sigma
                fit = hist.GetFunction("gaus")
                if fit:
                    graphs[f"{axis}_SIG"].SetPoint(idx, momentum_avg, fit.GetParameter(2))
                    graphs[f"{axis}_SIG"].SetPointError(idx, 0, fit.GetParError(2))

            # Plotting X and Y Hits at UBT
            c_hits.Clear()
            c_hits.Divide(2,1)
            c_hits.cd(1); h_x_ubt.SetTitle(f"UBT X Distribution - {os.path.basename(sim)};X [mm]; counts [{bin_size_x} mm/bin]"); h_x_ubt.Draw()
            c_hits.cd(2); h_y_ubt.SetTitle(f"UBT Y Distribution - {os.path.basename(sim)};Y [mm]; counts [{bin_size_y} mm/bin]"); h_y_ubt.Draw()
            c_hits.Print(pdf_name_hits) 
            
            # Plotting px, py, pz at UBT
            c_dir.Clear()
            c_dir.Divide(2,1)
            c_dir.cd(1); h_theta_x.SetTitle(fr"#theta_{{x}} distribution - {os.path.basename(sim)};#theta_{{x}} [mrad]; counts [{bin_size_theta_x} mrad/bin]"); h_theta_x.Draw()
            c_dir.cd(2); h_theta_y.SetTitle(fr"#theta_{{y}} distribution - {os.path.basename(sim)};#theta_{{y}} [mrad]; counts [{bin_size_theta_y} mrad/bin]"); h_theta_y.Draw()
            c_dir.Print(pdf_name_momentum) 
            
    c_hits.Print(f"{pdf_name_hits}]")
    c_dir.Print(f"{pdf_name_momentum}]")
    print("Data extraction complete!")

    # ===========================================
    # 3. Theoretical predictions
    # ===========================================

    p_total_raw = np.array(p_values, dtype=np.float64)
    x_he_raw = np.array(x_he_mean, dtype=np.float64)
    x_sbt_raw = np.array(x_sbt_mean, dtype=np.float64)

    # Sort to keep momentum and thicknesses cleanly aligned
    sort_idx = np.argsort(p_total_raw)
    p_total = p_total_raw[sort_idx]
    x_he_sorted = x_he_raw[sort_idx]
    x_sbt_sorted = x_sbt_raw[sort_idx]
    n_points = len(p_total)

    # Total mass thickness
    x_total_sorted = x_he_sorted + x_sbt_sorted

    #Transversing only helium
    if args.sbt is None:
        sig_theo = MC_scat.MC_scattering(p_total, x_total_sorted, A = 4.0026)

    else:
        sig_theo = MC_scat.MC_scattering(
        p=p_total, 
        x_total=x_total_sorted, 
        SBT=True, 
        x_He=x_he_sorted, 
        x_SBT=x_sbt_sorted, 
        A=4.0026, Z=2,       # Helium properties
        A_SBT=12.0, Z_SBT=6.0 # Scintillator properties
    )

    sig_theo_mrad = sig_theo * 1000.0
    
    # Estimate spatial spread using average distance or dynamic bounds
    distances_raw = np.array(distances, dtype=np.float64)
    distances_aligned = distances_raw[sort_idx]
    sig_theo_r = sig_theo * (distances_aligned * 10.0) / np.sqrt(3)

    gr_theory_angle = ROOT.TGraph(n_points, p_total, sig_theo_mrad)
    gr_theory_dist  = ROOT.TGraph(n_points, p_total, sig_theo_r)

    # ===========================================
    # 4. PLOTTING STATISTICS AND ENERGY DEPENDENCE
    # ===========================================
    def style_graph(gr, color, style, marker):
        gr.SetLineColor(color)
        gr.SetMarkerColor(color)
        gr.SetMarkerStyle(marker)
        gr.SetMarkerSize(1.2)
        gr.SetLineWidth(2)
        gr.SetLineStyle(style) 

    # Helper function for RMS and Sigma plots
    def plot_rms_sig(axis, gr_rms, gr_sig, title, output_dir, gr_theory = None):
        c = ROOT.TCanvas(f"c_{axis}", "", 800, 600)
        c.SetGrid()
        
        style_graph(gr_rms, ROOT.kBlue+1, 1, 20)
        style_graph(gr_sig, ROOT.kRed+1, 1, 21)

        # --- NEW: Calculate maximum Y value to prevent clipping ---
        max_y = ROOT.TMath.MaxElement(gr_rms.GetN(), gr_rms.GetY())
        if gr_theory:
            theory_max = ROOT.TMath.MaxElement(gr_theory.GetN(), gr_theory.GetY())
            max_y = max(max_y, theory_max)
            
        # Add 20% headroom to the top of the plot
        gr_rms.GetYaxis().SetRangeUser(0, max_y * 1.2)
        
        # Setup Fits
        fit_rms = ROOT.TF1(f"fit_rms_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_rms.SetParameters(0, 100, 1)
        fit_rms.SetLineColor(ROOT.kBlue+1)
        
        fit_sig = ROOT.TF1(f"fit_sig_{axis}", "[0] + [1]*pow(x,-[2])", 1, 100)
        fit_sig.SetParameters(0, 100, 1)
        fit_sig.SetLineColor(ROOT.kRed+1)
        
        gr_rms.Fit(fit_rms, "Q R S")
        gr_sig.Fit(fit_sig, "Q R S")
        
        gr_rms.SetTitle(title)
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

        # --- NEW: Overlay theoretical curve ---
        if gr_theory:
            gr_theory.SetLineColor(ROOT.kGreen+2)
            gr_theory.SetLineWidth(2)
            gr_theory.SetLineStyle(7) # 7 is a dashed line 
            gr_theory.Draw("C SAME")  # C draws a smooth curve between points

        stats_sig = gr_sig.FindObject("stats")
        if stats_sig:
            stats_sig.SetX1NDC(0.65); stats_sig.SetX2NDC(0.90)
            stats_sig.SetY1NDC(0.30); stats_sig.SetY2NDC(0.55)
            stats_sig.SetTextColor(ROOT.kRed+1)
            stats_sig.SetLineColor(ROOT.kRed+1)

        c.Update()

        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_rms, "RMS", "lep")
        leg.AddEntry(gr_sig, "Gaussian Sigma", "lep")

        if gr_theory:
            leg.AddEntry(gr_theory, "Theory (Highland)", "l")

        leg.SetBorderSize(0)
        leg.Draw()
        
        c.Modified()
        c.Update()
        c.SaveAs(os.path.join(output_dir, f"{axis}_STATS.png"))
        
        # Return elements to prevent PyROOT garbage collection wiping the canvas memory
        return c, fit_rms, fit_sig, leg

    # Helper function for Mean plots
    def plot_mean(axis, gr_mean, title, output_dir):
        c2 = ROOT.TCanvas(f"c2_{axis}", "", 900, 700)
        c2.SetGrid()
        
        style_graph(gr_mean, ROOT.kOrange+1, 1, 20)
        
        gr_mean.SetTitle(title)
        gr_mean.Draw("AP")
        c2.Update()

        stats_mean = gr_mean.FindObject("stats")
        if stats_mean:
            stats_mean.SetX1NDC(0.65); stats_mean.SetX2NDC(0.90)
            stats_mean.SetY1NDC(0.60); stats_mean.SetY2NDC(0.85)
            stats_mean.SetTextColor(ROOT.kOrange+1)
            stats_mean.SetLineColor(ROOT.kOrange+1)

        leg = ROOT.TLegend(0.20, 0.70, 0.50, 0.90)
        leg.AddEntry(gr_mean, "Mean", "lep")
        leg.SetBorderSize(0)
        leg.Draw()
        
        c2.Modified()
        c2.Update()
        c2.SaveAs(os.path.join(output_dir, f"{axis}_MEAN.png"))
        
        return c2, leg

    # --- EXECUTION ---
    # Store returned objects in a list to keep them alive in memory
    drawn_objects = []

    # 1. UBT Spatial Plots
    for axis in ["UBT_X", "UBT_Y"]:
        drawn_objects.append(plot_rms_sig(
            axis, graphs[f"{axis}_RMS"], graphs[f"{axis}_SIG"], 
            f"{axis} Spread Comparison;momentum [GeV];Spread [mm]", output_dir, gr_theory = gr_theory_dist
        ))
        drawn_objects.append(plot_mean(
            axis, graphs[f"{axis}_MEAN"], 
            f"{axis} Mean Spread;momentum [GeV];Spread [mm]", output_dir
        ))

    # 2. Angular Direction Plots
    for axis in ["theta_X", "theta_Y"]:
        drawn_objects.append(plot_rms_sig(
            axis, graphs[f"{axis}_RMS"], graphs[f"{axis}_SIG"], 
            fr"{axis} Direction comparison ;momentum [GeV];#theta_{axis[-1:]} [mrad]", output_dir, gr_theory = gr_theory_angle
        ))
        drawn_objects.append(plot_mean(
            axis, graphs[f"{axis}_MEAN"], 
            fr"{axis} Mean Direction ;momentum [GeV];#theta_{axis[-1:]} [mrad]", output_dir
        ))


if __name__ == "__main__":
    main()