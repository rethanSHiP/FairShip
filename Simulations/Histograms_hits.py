import argparse
import os
import glob
import uproot
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import awkward as ak


def main():
    parser = argparse.ArgumentParser(description="Propagate muons and plot deviations.")
    parser.add_argument("--field_file", default="../files/2025_02_12_SHiP_SpectrometerField_ECN3_MgB2.root", help="Path to magnet field file")
    parser.add_argument("--sim_file", default="energy_scan/sim_muon_50GeV.root", help="Path to simulation file")
    parser.add_argument("--z_center", type=float, default=8957.0, help="Z offset for field map")
    parser.add_argument("--UBT_z", type=float, default=3200.0, help="Target Z position")
    parser.add_argument("--step_size", type=float, default=-5.0, help="Step size for RK4")
    parser.add_argument("--tag", default="distributions", help="Filename for the plot")
    args = parser.parse_args()

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
                "MCTrack.fStartX", "MCTrack.fStartY" #UBT Truth position
                ], library="ak")
    
        # Create a boolean mask to ONLY look at hits belonging to the primary muon (TrackID == 0)
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

        # Exclude any rows with NaN values to ensure we only work with complete data
        valid_mask = ~np.isnan(x_sst_raw)
        if not np.any(valid_mask):
            print(f"Skipping {files}: 0 primary muons reached the SST.")
            # Safely exit the function if no muons made it
            return [], [], [], [], [], []

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

        q = -1.0
        dz = args.step_size
        target_z = args.UBT_z

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
            k2 = dz*get_derivatives_vec(x + 0.5*k1[0], y + 0.5*k1[1], z + 0.5*dz, px + 0.5*k1[2], py + 0.5*k1[3], pz2)

            # Stage 3 (at midpoint again)
            pz3 = np.sqrt(np.maximum(0, p_total**2 - (px + 0.5*k2[2])**2 - (py + 0.5*k2[3])**2))
            k3 = dz*get_derivatives_vec(x + 0.5*k2[0], y + 0.5*k2[1], z + 0.5*dz, px + 0.5*k2[2], py + 0.5*k2[3], pz3)
            
            # Stage 4 (at end)
            pz4 = np.sqrt(np.maximum(0, p_total**2 - (px + k3[2])**2 - (py + k3[3])**2))
            k4 = dz*get_derivatives_vec(x + k3[0], y + k3[1], z + dz, px + k3[2], py + k3[3], pz4)
            
            # Combine steps
            x +=  (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
            y +=  (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
            px += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
            py += (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6.0
            z += dz
            pz = np.sqrt(np.maximum(0, p_total**2 - px**2 - py**2))

        # Add results back to DataFrame
        ubt_X_mm = (x - mc_x) * 10
        ubt_Y_mm = (y - mc_y) * 10
        ubt_Z_cm = (z)
        sst_X_mm = x_sst_raw[valid_mask] * 10
        sst_Y_mm = y_sst_raw[valid_mask] * 10
        sst_Z_cm = z_sst_raw[valid_mask]

        return ubt_X_mm, ubt_Y_mm, ubt_Z_cm, sst_X_mm, sst_Y_mm, sst_Z_cm

    # ==========================================
    # 5. PLOTTING THE RESULTS
    # ==========================================

    sim = args.sim_file
    ubt_X_mm, ubt_Y_mm, ubt_Z_cm, sst_X_mm, sst_Y_mm, sst_Z_cm = Backward_Propagate_Muons(sim)

    output_dir = "histograms"
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to plot on an axis
    def plot_hist(ax, data, title, xlabel, color):
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Backward Propagation Summary Plots---------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_hist(axes[0], ubt_X_mm, "Distribution at UBT (X)", " X [mm]", 'blue')
    plot_hist(axes[1], ubt_Y_mm, "Distribution at UBT (Y)", " Y [mm]", 'red')

    # XY Correlation (2D Histogram)
    h = axes[2].hist2d(ubt_X_mm, ubt_Y_mm, bins=100, cmap='viridis')
    axes[2].set_title("Distribution at UBT (XY)")
    axes[2].set_xlabel("X [mm]")
    axes[2].set_ylabel("Y [mm]")
    fig.colorbar(h[3], ax=axes[2], label='Counts')

    plt.tight_layout()
    filename1 =  f"{args.tag}_UBT_Summary.png"
    savepath1 = os.path.join(output_dir,filename1)

    plt.savefig(savepath1, dpi=300)
    print(f"Summary plot saved as: {filename1}")

    # SST Summary Plots---------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    plot_hist(axes2[0], sst_X_mm, "Distribution at SST (X)", " X [mm]", 'blue')
    plot_hist(axes2[1], sst_Y_mm, "Distribution at SST (Y)", " Y [mm]", 'red')

    # XY Correlation (2D Histogram)
    h = axes2[2].hist2d(sst_X_mm, sst_Y_mm , bins=100, cmap='viridis')
    axes2[2].set_title("Distribution at SST (XY)")
    axes2[2].set_xlabel("X [mm]")
    axes2[2].set_ylabel("Y [mm]")
    fig2.colorbar(h[3], ax=axes2[2], label='Counts')

    plt.tight_layout()
    filename2 =  f"{args.tag}_SST_Summary.png"
    savepath2 = os.path.join(output_dir,filename2)

    plt.savefig(savepath2, dpi=300)
    print(f"Summary plot saved as: {filename2}")

    # B field Summary Plots---------------------
    tol = 1.0 
    df_center = df_field[(np.abs(df_field['x']) < tol) & (np.abs(df_field['y']) < tol)].sort_values('z')

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # 3. Plot each component
    ax3.plot(df_center['z'], df_center['Bx'], label='Bx [T]', color='red', linestyle=':', marker='o', markersize=4)
    ax3.plot(df_center['z'], df_center['By'], label='By [T]', color='green', linestyle='--', marker='^', markersize=4)
    ax3.plot(df_center['z'], df_center['Bz'], label='Bz [T]', color='blue', linestyle='-', marker='s', markersize=4)

    # 4. Styling
    ax3.set_title("Magnetic Field Profile along Central Z-axis")
    ax3.set_xlabel("Z-coordinate [mm]")
    ax3.set_ylabel("Magnetic Field Component [Tesla]")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    plt.tight_layout()
    filename3 =  f"{args.tag}_B_field.png"
    savepath3 = os.path.join(output_dir,filename3)

    plt.savefig(savepath3, dpi=300)
    print(f"Combined B-field plot saved to: {savepath3}")


    # ==========================================
    # 5. Safety checks
    # ==========================================
    # Backward Propagation Summary Plots---------------------
    fig4, axes4 = plt.subplots(1, 2, figsize=(18, 5))

    plot_hist(axes4[0], ubt_Z_cm, "Distribution at UBT (Z)", " Z [cm]", 'green')
    plot_hist(axes4[1], sst_Z_cm, "Distribution at SST (Z)", " Z [cm]", 'orange')

    axes4[0].ticklabel_format(useOffset=False, style='plain', axis='x')
    axes4[1].ticklabel_format(useOffset=False, style='plain', axis='x')

    plt.tight_layout()
    filename4 =  f"{args.tag}_Z_positions.png"
    savepath4 = os.path.join(output_dir,filename4)

    plt.savefig(savepath4, dpi=300)
    print(f"Summary plot saved as: {filename4}")
if __name__ == "__main__":
    main()