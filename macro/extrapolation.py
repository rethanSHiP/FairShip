import argparse
import os
import sys
import shutil
import uproot
import ROOT
import numpy as np
import pandas as pd
from array import array
from scipy.interpolate import NearestNDInterpolator

# Try importing FairShip environment safely
try:
    import shipRoot_conf
except ImportError:
    print("CRITICAL ERROR: FairShip environment not loaded.")
    print("Please run 'alienv enter FairShip/latest' first!")
    sys.exit(1)

# ================================================
# 1. Load the Magnetic field 
# ================================================
def load_magnetic_field(field_file, z_center):
    with uproot.open(field_file) as f:
        df_field = f["Data"].arrays(["x", "y", "z", "Bx", "By", "Bz"], library="pd")

    # Convert the local coordinates of the magnetic field to the global coordinates of the experiment
    df_field["z"] = df_field["z"] + z_center    

    bounds = {
        'x_min': df_field['x'].min(), 'x_max': df_field['x'].max(),
        'y_min': df_field['y'].min(), 'y_max': df_field['y'].max(),
        'z_min': df_field['z'].min(), 'z_max': df_field['z'].max()
    }

    # Create a SciPy 3D interpolator 
    points = df_field[['x', 'y', 'z']].values
    b_values = df_field[['Bx', 'By', 'Bz']].values
    b_field_map = NearestNDInterpolator(points, b_values)

    print("Magnetic Field Ready!")

    print("\n--- FIELD MAP BOUNDING BOX ---")
    print(f"X Bounds: {bounds['x_min']} to {bounds['x_max']}")
    print(f"Z Bounds: {bounds['z_min']} to {bounds['z_max']}")
    print("------------------------------\n")
    return b_field_map, bounds

# ================================================
# 2. Get derivatives for the RK4 algorithm 
# ================================================

def get_derivatives_vec(b_field_map, bounds, x, y, z, px, py, pz, q):
    Bx, By, Bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    mask = (x >= bounds['x_min']) & (x <= bounds['x_max']) & \
       (y >= bounds['y_min']) & (y <= bounds['y_max']) & \
       (z >= bounds['z_min']) & (z <= bounds['z_max'])

    if np.any(mask):
        coords = np.stack([x[mask], y[mask], z[mask]], axis=1)
        field = b_field_map(coords)
        Bx[mask], By[mask], Bz[mask] = field[:, 0], field[:, 1], field[:, 2]

    # Define the constant for the Lorentz force equation & return the derivatives    
    k = 0.0029979
    dx_dz = px / pz
    dy_dz = py / pz
    dpx_dz = k * q / pz * (py * Bz - pz * By)
    dpy_dz = k * q / pz * (pz * Bx - px *Bz)

    return np.array([dx_dz, dy_dz, dpx_dz, dpy_dz])

# ================================================
#            2. Get smeared positions 
# ================================================

def extract_smeared_data(smeared_file):
    print(f"Reading tracks from: {smeared_file}")
    f_in = ROOT.TFile.Open(smeared_file, "READ")
    tree = f_in.Get("ship_reco_sim")
    
    n_events = tree.GetEntries()
    print(f"Total events in chained Tree")

    sst_x = np.full(n_events, np.nan)
    sst_y = np.full(n_events, np.nan)
    sst_z = np.full(n_events, np.nan)
    sst_px = np.full(n_events, np.nan)
    sst_py = np.full(n_events, np.nan)
    sst_pz = np.full(n_events, np.nan)
    charge = np.full(n_events, np.nan)

    events_with_primary = 0
    for i in range(n_events):
        tree.GetEntry(i)

        tracks = getattr(tree,"FitTracks",None)
        tracks2mc = getattr(tree,"fitTrack2MC",None)


        if len(tracks) and len(tracks2mc) > 0:
            for track_idx, track in enumerate(tracks):            
                if track.getFitStatus().isFitConverged():
                    events_with_primary +=1
                    state = track.getFittedState()
                    pos, mom = state.getPos(), state.getMom()

                    sst_x[i], sst_y[i], sst_z[i] = pos.X(), pos.Y(), pos.Z()
                    sst_px[i], sst_py[i], sst_pz[i] = mom.X(), mom.Y(), mom.Z()
                    charge[i] = track.getFitStatus().getCharge()
                break

    f_in.Close()

    print("\n--- EXTRACTION DIAGNOSTICS ---")
    print(f"Events with CONVERGED primary muons: {events_with_primary} out of {n_events}")
    print("------------------------------\n")

    positions_sst = [sst_x, sst_y, sst_z]
    momentums_sst = [sst_px, sst_py, sst_pz]

    return positions_sst, momentums_sst, n_events, charge

# ================================================
# 4. Backward Propagation of Muons to UBT Function 
# ================================================
def Backward_Propagate_Muons(smeared_file, b_field_map, bounds, step_size, ubt_z):
    # Retriving smeared data
    positions, momentums, n_events, charge = extract_smeared_data(smeared_file)
    
    # Mask to only run heavy math on valid tracks
    valid_mask = ~np.isnan(positions[0])
    
    x = positions[0][valid_mask].copy()
    y = positions[1][valid_mask].copy()
    z = positions[2][valid_mask].copy()
    px = momentums[0][valid_mask].copy()
    py = momentums[1][valid_mask].copy()
    pz = momentums[2][valid_mask].copy()
    p_total = np.sqrt(px**2 + py**2 + pz**2)
    q = charge[valid_mask].copy()

    # Configurating the RK4 algorithm  
    dz = step_size
    
    # RK4 algorithm
    print(f"Propagating {len(x)} muons backwards...")
    while np.any(z > ubt_z):
        # Avoids overshooting the UBT
        dz = np.where(z <= ubt_z, 0.0, np.maximum(dz, ubt_z - z))
        
        # RK4 Math
        k1 = dz * get_derivatives_vec(b_field_map, bounds, x, y, z, px, py, pz, q)
        pz2 = np.sqrt(np.maximum(0, p_total**2 - (px + 0.5*k1[2])**2 - (py + 0.5*k1[3])**2))
        
        k2 = dz * get_derivatives_vec(b_field_map, bounds, x + 0.5*k1[0], y + 0.5*k1[1], z + 0.5*dz, px + 0.5*k1[2], py + 0.5*k1[3], pz2, q)
        pz3 = np.sqrt(np.maximum(0, p_total**2 - (px + 0.5*k2[2])**2 - (py + 0.5*k2[3])**2))
        
        k3 = dz * get_derivatives_vec(b_field_map, bounds, x + 0.5*k2[0], y + 0.5*k2[1], z + 0.5*dz, px + 0.5*k2[2], py + 0.5*k2[3], pz3, q)
        pz4 = np.sqrt(np.maximum(0, p_total**2 - (px + k3[2])**2 - (py + k3[3])**2))
        
        k4 = dz * get_derivatives_vec(b_field_map, bounds, x + k3[0], y + k3[1], z + dz, px + k3[2], py + k3[3], pz4, q)
        
        # Combine steps
        x += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
        y += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
        z += dz
        px += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
        py += (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6.0
        pz = np.sqrt(np.maximum(0, p_total**2 - px**2 - py**2))

    # Add results back to DataFrame
    final_x = np.full(n_events, np.nan)
    final_y = np.full(n_events, np.nan)
    final_z = np.full(n_events, np.nan)
    final_px = np.full(n_events, np.nan)
    final_py = np.full(n_events, np.nan)
    final_pz = np.full(n_events, np.nan)

    final_x[valid_mask] = x
    final_y[valid_mask] = y
    final_z[valid_mask] = z
    final_px[valid_mask] = px
    final_py[valid_mask] = py
    final_pz[valid_mask] = pz

    return final_x, final_y, final_z, final_px, final_py, final_pz, n_events

# ================================================
# 5. Update the ROOT file 
# ================================================

def update_root_file(input_file, output_file, pos_x, pos_y, pos_z, mom_px, mom_py, mom_pz, n_events):
    print(f"Creating exact copy of the smeared file at: {output_file}")
    shutil.copy(input_file, output_file)

    # Open the copied file in UPDATE mode so we can add new branches
    print("Appending new extrapolated branches...")
    f_out = ROOT.TFile(output_file, "UPDATE")
    tree = f_out.Get("ship_reco_sim")

    # Create Python arrays to hold the memory for the new branches
    ext_x = array('d', [0.0])
    ext_y = array('d', [0.0])
    ext_z = array('d', [0.0])
    ext_px = array('d', [0.0])
    ext_py = array('d', [0.0])
    ext_pz = array('d', [0.0])

    # Link the arrays to the new branches (Type 'D' is for C++ Double)
    b_x = tree.Branch("extrap_ubt_x", ext_x, "extrap_ubt_x/D")
    b_y = tree.Branch("extrap_ubt_y", ext_y, "extrap_ubt_y/D")
    b_z = tree.Branch("extrap_ubt_z", ext_z, "extrap_ubt_z/D")

    b_px = tree.Branch("extrap_ubt_px", ext_px, "extrap_ubt_px/D")
    b_py = tree.Branch("extrap_ubt_py", ext_py, "extrap_ubt_py/D")
    b_pz = tree.Branch("extrap_ubt_pz", ext_pz, "extrap_ubt_pz/D")

    # Loop through exactly once and Fill() only the new branches
    for i in range(n_events):
        tree.GetEntry(i)
        
        ext_x[0] = pos_x[i]
        ext_y[0] = pos_y[i]
        ext_z[0] = pos_z[i]

        ext_px[0] = mom_px[i]
        ext_py[0] = mom_py[i]
        ext_pz[0] = mom_pz[i]

        b_x.Fill()
        b_y.Fill()
        b_z.Fill()

        b_px.Fill()
        b_py.Fill()
        b_pz.Fill()

    # Overwrite the old tree header with the new one containing the branches
    tree.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    print("Success! The branches have been safely appended.")
    
# ==========================================
# 5. MAIN LOOP
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--field_file", default="files/2025_02_12_SHiP_SpectrometerField_ECN3_MgB2.root", help="Path to magnet field file")
    parser.add_argument("--z_center", type=float, default=8957.0, help="Z offset for field map")
    parser.add_argument("--UBT_z", type=float, default=3272.00, help="Target Z position")
    parser.add_argument("--step_size", type=float, default=-5.0, help="Step size for RK4")
    args = parser.parse_args()

    # Set up paths
    in_file = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/smeared/chained_smeared.root"
    out_dir = "/afs/cern.ch/work/r/rethan/public/FairShip/cuda_muons_simulations/extrapolated"
    out_file = os.path.join(out_dir,"extrapolated.root")
    os.makedirs(out_dir, exist_ok=True)

    # Load the magnetic field
    b_field_map, b_field_bounds = load_magnetic_field(args.field_file, args.z_center)

    # Extrapolating muons to the UBT
    x,y,z,px,py,pz,n = Backward_Propagate_Muons(in_file, b_field_map, b_field_bounds, args.step_size, args.UBT_z)

    # Storing the new results
    update_root_file(in_file, out_file, x, y, z, px, py, pz, n)


if __name__ == "__main__":
    main()