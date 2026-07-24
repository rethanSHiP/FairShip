import argparse
import os
import glob
import uproot
import ROOT
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import awkward as ak


def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--field_file", default="../files/2025_02_12_SHiP_SpectrometerField_ECN3_MgB2.root", help="Path to magnet field file")
    parser.add_argument("--z_center", type=float, default=8957.0, help="Z offset for field map")
    parser.add_argument("--UBT_z", type=float, default=3200.0, help="Target Z position")
    parser.add_argument("--step_size", type=float, default=-5.0, help="Step size for RK4")
    parser.add_argument("--in_dir", default="energy_scan", help="Path to simulation files") 
    parser.add_argument("--out_dir", type=str, default=None, help="name the output directory")
    args = parser.parse_args()

    if args.out_dir is None: args.out_dir = args.in_dir

    # ================================================
    # 1. LOAD MAGNETIC FIELD & Get Derivatives for RK4
    # ================================================
    
    # Magnetic field

    with uproot.open(args.field_file) as f:
        df_field = f["Data"].arrays(["x", "y", "z", "Bx", "By", "Bz"], library="pd")

    # Convert the local coordinates of the magnetic field to the global coordinates of the experiment
    z_center = args.z_center
    df_field["z"] = df_field["z"] + z_center    

    x_min, x_max = df_field['x'].min(), df_field['x'].max()
    y_min, y_max = df_field['y'].min(), df_field['y'].max()
    z_min, z_max = df_field['z'].min(), df_field['z'].max()

    # Create a SciPy 3D interpolator 
    points = df_field[['x', 'y', 'z']].values
    b_values = df_field[['Bx', 'By', 'Bz']].values
    b_field_map = NearestNDInterpolator(points, b_values)

    print("Magnetic Field Ready!")

    # Derivatives for RK4 integration

    def get_derivatives_vec(x, y, z, px, py, pz, q=-1.0):
        Bx, By, Bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)

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
    # 3. Backward Propagation of Muons to UBT Function 
    # ================================================
    def Backward_Propagate_Muons(files):
        with uproot.open(files) as sim_file:
            tree = sim_file["cbmsim"]

            # Convert TTree into akward array for easier manipulation
            ak_muons = tree.arrays([
                "strawtubesPoint.fTrackID", #ID 
                "strawtubesPoint.fX", "strawtubesPoint.fY", "strawtubesPoint.fZ", #SST position
                "strawtubesPoint.fPx", "strawtubesPoint.fPy", "strawtubesPoint.fPz", #SST momentum
                "MCTrack.fStartX", "MCTrack.fStartY", "MCTrack.fStartZ", #UBT Truth position
                "MCTrack.fPx","MCTrack.fPy","MCTrack.fPz" #UBT Truth momentum
                ], library="ak")

        is_primary_hit = ak_muons["strawtubesPoint.fTrackID"] == 0
        # Apply the mask ONLY to the strawtubes (SST) hit arrays
        x_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fX"][is_primary_hit]), np.nan))
        y_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fY"][is_primary_hit]), np.nan))
        z_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fZ"][is_primary_hit]), np.nan))
        
        px_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fPx"][is_primary_hit]), np.nan))
        py_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fPy"][is_primary_hit]), np.nan))
        pz_sst_raw = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["strawtubesPoint.fPz"][is_primary_hit]), np.nan))
        
        mc_start_x = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fStartX"]), np.nan))
        mc_start_y = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fStartY"]), np.nan))
        mc_start_z = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fStartZ"]), np.nan))

        mc_px_ubt = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fPx"]), np.nan))
        mc_py_ubt = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fPy"]), np.nan))
        mc_pz_ubt = ak.to_numpy(ak.fill_none(ak.firsts(ak_muons["MCTrack.fPz"]), np.nan))

        # Exclude any rows with NaN values to ensure we only work with complete data
        valid_mask = ~np.isnan(x_sst_raw)
        if not np.any(valid_mask):
            print(f"Skipping {files}: 0 primary muons reached the SST.")
            return [], [], [], [], []

        # ==========================================
        # 4. PROPAGATE MUONS BACKWARDS TO UBT
        # ==========================================

        # Grab the initial state at the SST
        x = x_sst_raw[valid_mask].copy()
        y = y_sst_raw[valid_mask].copy()
        z = z_sst_raw[valid_mask].copy()
        px = px_sst_raw[valid_mask].copy()
        py = py_sst_raw[valid_mask].copy()
        pz = pz_sst_raw[valid_mask].copy()
        p_total = np.sqrt(px**2 + py**2 + pz**2)
        
        mc_x = mc_start_x[valid_mask]
        mc_y = mc_start_y[valid_mask]
        mc_z = mc_start_z[valid_mask]

        mc_px = mc_px_ubt[valid_mask]
        mc_py = mc_py_ubt[valid_mask]   
        mc_pz = mc_pz_ubt[valid_mask]        

        q = -1.0
        dz = args.step_size
        target_z = args.UBT_z

        # Obtain the matter density transversed
        x_he_accum = np.zeros(len(x))
        x_sbt_accum = np.zeros(len(x))
        distances = np.zeros(len(x))
        
        # Densities (g/cm^3)
        rho_he = 1.675e-4 
        rho_sbt = 1.032

        print(f"Propagating {len(x)} muons backwards...")
        # Propagate the muons backwards until they reach the UBT
        while np.any(z > args.UBT_z):

            # Avoids overshooting the UBT
            dz = np.where(z <= args.UBT_z, 0.0, np.maximum(args.step_size, args.UBT_z - z))
            
            # We have four differential equations to solve (k = [x,y,px,py])
            # Let's compute each stage for each variable

            # Stage 1 (initial)
            k1 = dz*get_derivatives_vec(x, y, z, px, py, pz, q = q)

            # Stage 2(middle point)
            pz2 = np.sqrt(np.maximum(0, p_total**2 - (px + 0.5*k1[2])**2 - (py + 0.5*k1[3])**2))
            k2 = dz*get_derivatives_vec(x + 0.5*k1[0], y + 0.5*k1[1], z + 0.5*dz, px + 0.5*k1[2], py + 0.5*k1[3], pz2, q = q)

            # Stage 3 (at midpoint again)
            pz3 = np.sqrt(np.maximum(0, p_total**2 - (px + 0.5*k2[2])**2 - (py + 0.5*k2[3])**2))
            k3 = dz*get_derivatives_vec(x + 0.5*k2[0], y + 0.5*k2[1], z + 0.5*dz, px + 0.5*k2[2], py + 0.5*k2[3], pz3, q = q)
            
            # Stage 4 (at end)
            pz4 = np.sqrt(np.maximum(0, p_total**2 - (px + k3[2])**2 - (py + k3[3])**2))
            k4 = dz*get_derivatives_vec(x + k3[0], y + k3[1], z + dz, px + k3[2], py + k3[3], pz4, q = q)
            
            # Combine steps
            dx  =  (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
            dy  =  (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
            dz  = dz
            ds = np.sqrt(dx**2 + dy**2 + dz**2)
            
            dpx = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
            dpy = (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6.0
            
            x += dx
            y += dy
            z += dz
            px += dpx
            py += dpy
            pz = np.sqrt(np.maximum(0, p_total**2 - px**2 - py**2))
            distances += ds

            # Defining the limits of the SBT
            Z_start, Z_end = 3312.0, 8312.0

            Y_inner_limit = 270.0 + ((600.0 - 270.0) / (Z_end - Z_start)) * (z - Z_start)
            Y_outer_limit = Y_inner_limit + 20.0

            X_inner_limit = 100.0 + ((400.0 - 100.0) / (Z_end - Z_start)) * (z - Z_start)
            X_outer_limit = X_inner_limit + 20.0

            abs_y = np.abs(y)
            abs_x = np.abs(x)

            in_decay_volume = (z >= Z_start) & (z <= Z_end)
            in_helium = in_decay_volume & (abs_y <= Y_inner_limit) & (abs_x <= X_inner_limit)
            in_sbt = in_decay_volume & (abs_y > Y_inner_limit) & (abs_y <= Y_outer_limit) & (abs_x > X_inner_limit) & (abs_x <= X_outer_limit)
            
            x_he_accum += np.where(in_helium, ds * rho_he, 0.0)
            x_sbt_accum += np.where(in_sbt, ds * rho_sbt, 0.0)

        # Add results back to DataFrame
        ubt_positions_reco = [x,y,z]
        ubt_positions_true = [mc_x, mc_y, mc_z]

        ubt_momentum_reco = [px, py, pz]
        ubt_momentum_true = [mc_px, mc_py, mc_pz]

        return ubt_positions_reco, ubt_positions_true, ubt_momentum_reco, ubt_momentum_true, x_he_accum, x_sbt_accum, distances


    def Save_Propagated_Data(outputfile, pos_reco, pos_true, mom_reco, mom_true, x_he, x_sbt,dist):
        with uproot.recreate(outputfile) as f: 
            f["UBT_Muons"] = {
                "X": pos_reco[0], "Y": pos_reco[1], "Z": pos_reco[2],
                "X_true": pos_true[0], "Y_true": pos_true[1], "Z_true": pos_true[2],
                "PX": mom_reco[0], "PY": mom_reco[1], "PZ": mom_reco[2],
                "PX_true": mom_true[0], "PY_true": mom_true[1], "PZ_true": mom_true[2],
                "x_He": x_he, "x_SBT": x_sbt, "distances":dist
                }

    # ==========================================
    # 5. MAIN LOOP
    # ==========================================
    simulation_paths = glob.glob(os.path.join(args.in_dir, "sim_muon_*.root"))
    
    output_dir = args.out_dir + "_reco"
    os.makedirs(output_dir, exist_ok=True)
    
    for i,sim in enumerate(simulation_paths): 
        print(f"Processing simulation: {sim})")
        positions, true_positions, momentum, true_momentum, x_he, x_sbt, dist = Backward_Propagate_Muons(sim)
        if len(positions[0]) == 0: continue

        name = os.path.basename(sim)
        output = os.path.join(output_dir, f"B_{name}")
        
        Save_Propagated_Data(output,positions, true_positions, momentum, true_momentum, x_he, x_sbt, dist)
    print("New proccessed files created")

if __name__ == "__main__":
    main()