import os
import sys
import numpy as np
import shutil
import ROOT

try:
    import shipRoot_conf
except ImportError:
    print("CRITICAL ERROR: FairShip environment not loaded.")
    print("Please run 'alienv enter FairShip/latest' first!")
    sys.exit(1)

def inject_poisson_noise(input_file, output_file, occupancy_percent=1.0):
    print(f"Creating exact copy of the source file at: {output_file}")
    shutil.copy(input_file, output_file)

    print("Opening file to inject background noise...")
    f_out = ROOT.TFile(output_file, "UPDATE")
    tree = f_out.Get("ship_reco_sim")
    n_events = tree.GetEntries()

    # 1. Physics Parameters for the UBT
    x_min, x_max = -220.0, 220.0
    y_min, y_max = -320.0, 320.0
    detector_area_cm2 = (x_max - x_min) * (y_max - y_min)
    lambda_bkg = detector_area_cm2 * (occupancy_percent / 100.0)

    print(f"Detector Area: {detector_area_cm2:,.0f} cm^2")
    print(f"Target Occupancy: {occupancy_percent}% -> Average {lambda_bkg:.0f} hits/event")

    # 2. Pre-calculate the Poisson distribution for speed
    hits_per_event = np.random.poisson(lambda_bkg, n_events)
    print(f"Injecting a total of {np.sum(hits_per_event):,.0f} background hits...")

    # 3. Create C++ std::vectors to hold the variable-length lists of hits
    vec_x = ROOT.std.vector('double')()
    vec_y = ROOT.std.vector('double')()
    vec_z = ROOT.std.vector('double')()
    vec_px = ROOT.std.vector('double')()
    vec_py = ROOT.std.vector('double')()
    vec_pz = ROOT.std.vector('double')()

    # 4. Link the vectors to the new branches
    b_x = tree.Branch("bkg_ubt_x", vec_x)
    b_y = tree.Branch("bkg_ubt_y", vec_y)
    b_z = tree.Branch("bkg_ubt_z", vec_z)
    b_px = tree.Branch("bkg_ubt_px", vec_px)
    b_py = tree.Branch("bkg_ubt_py", vec_py)
    b_pz = tree.Branch("bkg_ubt_pz", vec_pz)

    # 5. Loop through the events and fill the vectors
    for i in range(n_events):
        tree.GetEntry(i)
        
        # Clear the vectors for the new event!
        vec_x.clear(); vec_y.clear(); vec_z.clear()
        vec_px.clear(); vec_py.clear(); vec_pz.clear()

        n_hits = hits_per_event[i]
        if n_hits > 0:
            # Instantly generate all random coordinates for this specific event
            x = np.random.uniform(x_min, x_max, n_hits)
            y = np.random.uniform(y_min, y_max, n_hits)

            px = np.random.normal(0.0, 0.5, n_hits)
            py = np.random.normal(0.0, 0.5, n_hits)
            pz = np.random.exponential(scale=5.0, size=n_hits)
            
            # Push them into the C++ vectors
            for x, y, px, py, pz in zip(x, y, px, py, pz):
                vec_x.push_back(x)
                vec_y.push_back(y)
                vec_z.push_back(3279.57)
                vec_px.push_back(px)
                vec_py.push_back(py)
                vec_pz.push_back(pz)

        # Fill ONLY the new branches for this row
        b_x.Fill(); b_y.Fill(); b_z.Fill()
        b_px.Fill(); b_py.Fill(); b_pz.Fill()

        if i > 0 and i % 5000 == 0:
            print(f"  ... processed {i} / {n_events} events")

    # Overwrite the old tree header to lock in the new branches
    tree.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    print("Success! Background noise successfully integrated into the ROOT file.")

def search_area_x(p):
    return 0.03198 + 41.54*p**(-1.007)

def search_area_y(p):
    return 0.03572 + 40.98*p**(-1.004)

def search_area_px(p):
    return 0.0006186 + 1.326*p**(-0.02851)

def search_area_py(p):
    return 0.0006186 + 1.326*p**(-0.02851)

def chi_squared(smeared_hit, try_hit, p):
    sig_x, sig_y = search_area_x(p), search_area_y(p)
    sig_theta_x, sig_theta_y = search_area_px(p), search_area_py(p)

    delta_x_mm = (smeared_hit[0] - try_hit[0])* 10
    delta_y_mm = (smeared_hit[1] - try_hit[1])* 10

    px_s, py_s, pz_s = smeared_hit[3], smeared_hit[4], smeared_hit[5]
    theta_x_s = (px_s / pz_s) if pz_s != 0 else 0
    theta_y_s = (py_s / pz_s) if pz_s != 0 else 0

    # Reconstruct Pz for the try_hit (Truth or Background)
    px_t, py_t, pz_t = try_hit[3], try_hit[4], try_hit[5]
    theta_x_t = (px_t / pz_t) if pz_t != 0 else 0
    theta_y_t = (py_t / pz_t) if pz_t != 0 else 0

    # Relative angle calculation in mrad (same as your tan_rel function)
    delta_theta_x_mrad = np.arctan((theta_x_s - theta_x_t) / (1 + theta_x_s * theta_x_t)) * 1000.0
    delta_theta_y_mrad = np.arctan((theta_y_s - theta_y_t) / (1 + theta_y_s * theta_y_t)) * 1000.0

    # 4. Calculate the true Mahalanobis Distance / Chi-Squared
    chi2 = (delta_x_mm / sig_x)**2 + \
           (delta_y_mm / sig_y)**2 + \
           (delta_theta_x_mrad / sig_theta_x)**2 + \
           (delta_theta_y_mrad / sig_theta_y)**2
    return chi2

def matching_hits(): 
    detector_info = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/extrapolated/extrapolated_with_noise.root"
    truth_info = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/filtered_files/chained_filtered_files.root"
    hits_dir = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/extrapolated/hist"

    f_detector = ROOT.TFile.Open(detector_info, "READ")
    f_truth = ROOT.TFile.Open(truth_info, "READ")

    t_detector = f_detector.Get("ship_reco_sim")
    t_truth = f_truth.Get("cbmsim")

    correctly, wrongly, lost = 0,0,0
    CHI2_CUTOFF = 13.3 + 40

    n_events = t_truth.GetEntries()
    for i in range(n_events):
        t_detector.GetEntry(i)
        t_truth.GetEntry(i)

        if ~np.isnan(t_detector.extrap_ubt_x) and t_detector.extrap_ubt_x != -999.0:
            smeared_hit = [t_detector.extrap_ubt_x, t_detector.extrap_ubt_y,t_detector.extrap_ubt_z,
                          t_detector.extrap_ubt_px, t_detector.extrap_ubt_py, t_detector.extrap_ubt_pz]
            p_smeared = np.sqrt(t_detector.extrap_ubt_px**2 + t_detector.extrap_ubt_py**2 + t_detector.extrap_ubt_pz**2)

            valid_truth_hit = None
            max_z = -9999.0
            for hit in t_truth.UpstreamTaggerPoint:
                if hit.GetTrackID() == 0 and hit.GetZ() > max_z:
                    max_z = hit.GetZ()
                    valid_truth_hit = hit
            
            if not valid_truth_hit:
                continue
            try:
                truth_hit = [valid_truth_hit.GetX(), valid_truth_hit.GetY(), valid_truth_hit.GetZ(),
                             valid_truth_hit.GetPx(), valid_truth_hit.GetPy(), valid_truth_hit.GetPz()]
            except AttributeError:
                truth_hit = [valid_truth_hit.fX, valid_truth_hit.fY, valid_truth_hit.fZ,
                             valid_truth_hit.fPx, valid_truth_hit.fPy, valid_truth_hit.fPz]
            
            noise_x = list(t_detector.bkg_ubt_x)
            noise_y = list(t_detector.bkg_ubt_y)
            noise_z = list(t_detector.bkg_ubt_z)
            noise_px = list(t_detector.bkg_ubt_px)
            noise_py = list(t_detector.bkg_ubt_py)
            noise_pz = list(t_detector.bkg_ubt_pz)


            true_chi = chi_squared(smeared_hit,truth_hit, p_smeared)
            if i < 5:
                print(f"\n--- EVENT {i} DIAGNOSTICS ---")
                print(f"Momentum: {p_smeared:.2f} GeV")
                print(f"Sigmas -> X: {search_area_x(p_smeared):.2f} mm | Theta X: {search_area_px(p_smeared):.2f} mrad")
                print(f"Z-Planes -> Extrapolated Z: {smeared_hit[2]:.2f} | Truth Z: {truth_hit[2]:.2f}")
                
                delta_x_mm = (smeared_hit[0] - truth_hit[0]) * 10
                delta_y_mm = (smeared_hit[1] - truth_hit[1]) * 10
                print(f"X Drift -> Extrap: {smeared_hit[0]:.2f} cm | Truth: {truth_hit[0]:.2f} cm | Delta: {delta_x_mm:.2f} mm")
                print(f"Y Drift -> Extrap: {smeared_hit[1]:.2f} cm | Truth: {truth_hit[1]:.2f} cm | Delta: {delta_y_mm:.2f} mm")
                print(f"Px -> Extrap: {smeared_hit[3]:.4f} GeV | Truth: {truth_hit[3]:.4f} GeV")
                print(f"Py -> Extrap: {smeared_hit[4]:.4f} GeV | Truth: {truth_hit[4]:.4f} GeV")
                
                print(f"TOTAL TRUE CHI2: {true_chi:.2f}")
            bkg_chi = [chi_squared(smeared_hit,[noise_x[k],noise_y[k],noise_z[k],noise_px[k],noise_py[k],noise_pz[k]], p_smeared) for k in range(len(noise_x))]

            min_bkg_chi = min(bkg_chi) if bkg_chi else float('inf')

            if true_chi < min_bkg_chi:# and true_chi < CHI2_CUTOFF: 
                correctly += 1

            elif min_bkg_chi < true_chi:# and min_bkg_chi < CHI2_CUTOFF: 
                wrongly += 1
            #else:
                # If neither the true hit nor the background hit is inside the search area, the muon is officially "Lost"
               #lost += 1

    total_valid = correctly + wrongly + lost
    print("--- MATCHING RESULTS ---")
    print(f"Total Valid Extrapolations: {total_valid}")
    print(f"Correctly identified: {correctly}")
    print(f"Wrongly identified (False Positives): {wrongly}")
    print(f"Lost / Unmatched: {lost}")
    print(f"Accuracy: {(correctly/total_valid)*100:.2f}%")
        
def main():
    shipRoot_conf.configure()

    in_dir = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/extrapolated"
    in_file = os.path.join(in_dir, "extrapolated.root")
    out_file = os.path.join(in_dir, "extrapolated_with_noise.root")

    #inject_poisson_noise(in_file, out_file, occupancy_percent=1.0)
    matching_hits()

if __name__ == "__main__":
    main()