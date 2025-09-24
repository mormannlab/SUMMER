## functions for running difference in curves test - 28.12.22 
## e.g., cluster permutation test. 
## refs:
#### https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/
#### https://www.nature.com/articles/s41467-021-26327-3#Sec15
#### https://www.sciencedirect.com/science/article/abs/pii/S0165027007001707?via%3Dihub

import seaborn as sns
import matplotlib.pyplot as plt

import random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind

from scipy.io import savemat, loadmat

clusteralpha = 0.005
perm_alpha = 0.05

def parse_dataLabels(df, pvals_ind, alpha=0.001, return_inds=False):
    """
    parse a pandas dataframe of activity by its pvalues 
    into condition A (significant) and condition B (not sig) according 
    to the prescribed alpha level. 

    Args:
        df (pd.DataFrame): DataFrame inherited from regionXlabel_wrapper
        pvals_ind (np.array): pvalues corresponding to each unit (row) in df
        alpha (float, optional): alpha level for significance. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    # parse data + labels
    condA_inds = pvals_ind[pvals_ind <= alpha].index
    condB_inds = pvals_ind[pvals_ind > alpha].index

    condA = df.loc[condA_inds]
    condB = df.loc[condB_inds]
    
    if return_inds:
        return condA, condB, condA_inds, condB_inds

    else: 
        return condA, condB
    
def calc_testStat(condA, condB):
    """
    Calculate the test stat between condition A and condition B
    for each bin of activity. Returns the stats with dims (1, nm_bins), 
    with corresponding pvalues (1, nm_bins).

    Args:
        condA (pd.DataFrame): activity for condition A units    
        condB (pd.DataFrame): activity for condition B units

    Returns:
        _type_: _description_
    """
    # doesn't actually matter, but just for stability

    # condA.drop(columns=["patient_id", "unit_id"])
    # condB.drop(columns=["patient_id", "unit_id"])

    bin_ids = condA.columns

    gt_stats = np.empty(len(bin_ids))
    gt_pvals = np.empty(len(bin_ids))

    for i, ind in enumerate(bin_ids):
        #stat, pval = kruskal(condA[ind], condB[ind])
        stat, pval = ttest_ind(condA[ind], condB[ind])
        #print(round(stat, 2), round(pval,2))
        gt_stats[i] = (np.abs(stat))
        gt_pvals[i] = (pval)
        
    return gt_stats, gt_pvals

def cluster_curves(gt_stats, gt_pvals, clusteralpha=0.005, mode="all", verbose=False):
    """
    following recipe from https://doi.org/10.1016/j.jneumeth.2007.03.024
    
    note: currently using kruskal-wallace test, which is one-sided, so no accounting here for 
        absolute values of the test statistic bc all are positive.
    """
    # step 1: For every sample, compare the MEG-signal on the two types of trials (semantically congruent versus semantically incon-
    # gruent sentence endings) by means of a t-value (or some other number that quantifies the effect at this sample).
    # --> step already done, produce gt_stats, gt_pvals
    
    #step 2: Select all samples whose t-value is larger than some threshold.
    
    #corrected_alpha = alpha / len(gt_stats)
    
    binary_pvals = [1 if x <= clusteralpha else 0 for x in gt_pvals]

    #step 3: Cluster the selected samples in connected sets on the basis of temporal adjacency
    clus_collection = {}
    clus_id = 0
    clus_inds = []
    clus_keeper = 0

    for i, val in enumerate(binary_pvals):
        if val == 1:
            if clus_keeper == 0:
                clus_keeper = 1 # switch the flag on
                clus_inds.append(i)
            elif clus_keeper == 1:
                clus_inds.append(i)
        if val == 0:
            if clus_keeper == 1:
                clus_keeper = 0 # switch the flag off
                clus_collection[clus_id] = clus_inds
                clus_inds = []
                clus_id += 1
            elif clus_keeper == 0:
                continue
    
    if len(clus_collection.keys()) == 0:
        cluster_sum = 0
        cluster_inds = []
        
        if mode == "maxsum":
            return cluster_sum, cluster_inds
        elif mode == "all":
            return [], [], []
        
    else:
        
        if mode == "maxsum":
            ## continuing with algo as specified in the original paper
            
            #step 4: Calculate cluster-level statistics by taking the sum of the t-values within a cluster
            sums = []
            for clus_id in clus_collection.keys():
                if verbose:
                    print(clus_id)
                    print(sum(gt_stats[clus_collection[clus_id]]))
                sums.append(sum(gt_stats[clus_collection[clus_id]]))

            #step 5: Take the largest of the cluster-level statistics.
            largest_cluster = sums.index(max(sums))
            cluster_inds = clus_collection[largest_cluster]
            cluster_sum = sum(gt_stats[clus_collection[largest_cluster]])

            return cluster_sum, cluster_inds

        elif mode == "all":
            ## diverging to algo as is in group matlab code, perm_multcompare+perm_ttest
            #step 4: Calculate cluster-level statistics by taking the sum of the t-values within a cluster
            sums = []
            for clus_id in clus_collection.keys():
                if verbose:
                    print(clus_id)
                    print(sum(gt_stats[clus_collection[clus_id]]))
                sums.append(sum(gt_stats[clus_collection[clus_id]]))
            
            return sums, clus_collection.keys(), [clus_collection[k] for k in clus_collection.keys()]

def generate_permutations(df, condA_inds, condB_inds, nm_perms=1000):
    # generate perm indices, such that they can be slotted into the 
    # pipeline at calc_testStat()

    A_list = []
    B_list = []

    while nm_perms != 0:
        nm_A = len(condA_inds)
        condA_inds_ = random.sample(range(min(df.index), max(df.index)), nm_A)
        condB_inds_ = list(df.index)
        _ = [condB_inds_.remove(A) for A in condA_inds_]
        A_list.append(condA_inds_)
        B_list.append(condB_inds_)
        nm_perms = nm_perms - 1

    A_list = np.array(A_list)
    B_list = np.array(B_list)
    
    return A_list, B_list

def df_formatter2(condA_, condB_, name1, name2):
    
    bin_id = []
    val = []
    cat = []

    for row in condA_.iterrows():
        tt = row[1:]
        for _, t in enumerate(tt):
            for i, v in enumerate(t):
                bin_id.append(i)
                val.append(v)
                cat.append(name1)

    for row in condB_.iterrows():
        tt = row[1:]
        for _, t in enumerate(tt):
            for i, v in enumerate(t):
                bin_id.append(i)
                val.append(v)
                cat.append(name2)

    l_df = pd.DataFrame({"bin":bin_id, "FR":val, "cond":cat})
    return l_df


def permute_scores(df, A_list, B_list, condA_inds=None, condB_inds=None):
    sc_collection = []

    for perm_ind in range(0, len(A_list)):
        condA_ = df.loc[A_list[perm_ind]]
        condB_ = df.loc[B_list[perm_ind]]
        
        if condA_inds:
            name1 = f"unit overlap: {round(len(np.intersect1d(A_list[perm_ind], condA_inds)) / len(condA_inds), 2)}"
            name2 = f"unit overlap: {round(len(np.intersect1d(B_list[perm_ind], condB_inds)) / len(condB_inds), 2)}"

            d = df_formatter2(condA_, condB_, name1, name2)
            fig,ax = plt.subplots(1,1, figsize=(4,2))
            sns.lineplot(data=d, x="bin", y="FR", hue="cond", ax=ax, palette=["black", 'tab:orange'])
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
            plt.show()

        gt_stats_, gt_pvals_ = calc_testStat(condA_, condB_)

        cluster_score_, _ = cluster_curves(gt_stats_, gt_pvals_, mode="maxsum")
        sc_collection.append(cluster_score_)
    
    return np.array(sc_collection)

def get_cluster_pvals(sc_collection, clus_ids, clus_stats, nm_perms=1000):
    
    clus_pvals = []

    for i in clus_ids:

        tstat = clus_stats[i]
        print(tstat)
        if tstat > 0: # pos
            perm_dist = np.array(sorted(sc_collection, reverse=True))
        if tstat < 0: # pos
            perm_dist = np.array(sorted(sc_collection, reverse=False))

        clus_test = np.abs(perm_dist - clus_stats[i])

        val = np.min(clus_test)
        rank = np.where(clus_test == val)[0][0]

        clus_pvals.append(rank / nm_perms)
    return clus_pvals

def sig_marker(pval):
    if pval <= 0.05 and pval > 0.01:
        marker = "*"
    elif pval <= 0.01 and pval > 0.001:
        marker = "**"
    elif pval <= 0.001:
        marker = "***"

    return marker

def load_psth_data(label_name, region, key, results_dir):

    filename = f"{label_name}_{region}_key{key}.npy"
 
    d = np.load(Path(results_dir, filename), allow_pickle=True).item()
    df = d["df"]
    lineplot_df = d["lineplot_df"]
    total = d["total"] 
    start_ns = d["start_ns"]
    ct_sig_units = d["ct_sig_units"]
    pval_inds = d["pval_inds"]
    
    return df, lineplot_df, total, start_ns, ct_sig_units, pval_inds
