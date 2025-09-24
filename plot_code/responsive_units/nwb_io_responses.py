

from pathlib import Path
import numpy as np


def load_psth_data(label_name, region, key, results_dir):
    """
    Example usage:
    df_og, lineplot_df_og, total_og, start_ns_og, ct_sig_units_og, pval_inds_og = load_psth_data("camera-cuts", "PHC", 1, "/home/al/Documents/phd/code/dhv_dataset/plot_code/responsive_units/results/psth")
    phc_units = np.load("/home/al/Documents/phd/code/dhv_dataset/plot_code/responsive_units/results/phc_units.npy")
    """
    filename = f"{label_name}_{region}_key{key}.npy"
 
    d = np.load(Path(results_dir, filename), allow_pickle=True).item()
    df = d["df"]
    lineplot_df = d["lineplot_df"]
    total = d["total"] 
    start_ns = d["start_ns"]
    ct_sig_units = d["ct_sig_units"]
    pval_inds = d["pval_inds"]
    
    return df, lineplot_df, total, start_ns, ct_sig_units, pval_inds