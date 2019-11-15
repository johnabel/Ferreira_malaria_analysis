# -*- coding: utf-8 -*-
"""
Created on Mon July 23 2018

@author: john abel
python 3.7
"""

# imports the needed python packages
from collections import Counter
import pickle

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy import stats
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import gridspec

from local_imports import PlotOptions as plo
from local_imports import Processing as pro

#
#     Define functions
#

def cluster_expression_data(dataset, name, rhythmicity_data, cluster_basis='zscore', color_thresh=0.85, link_method='complete'):
    """
    performs the clustering of the data.
    returns a dictionary of outputs
    """
    ds_ = dataset
    # get values
    val_ = ds_.values

    # zscore or norm data
    ts_ = np.floor(np.array(ds_.columns).astype(float))
    zval_ = np.nan_to_num(stats.zscore(val_, axis=1)) # nan to 0
    nval_ = (val_.T/(np.mean(val_, axis=1)+1E-15)).T
    rval_ = val_
    lab_ = ds_.index.tolist()

    # handle the replicates.
    # take mean at each point of the zscore or of the normed value
    uts_ = np.sort(np.unique(ts_))  # unique values
    mzval_ = []  # mean z-score values
    mnval_ = [] # mean normed value
    mrval_ = [] # mean raw value
    for ti in uts_:
        tloc = np.where(ts_ == ti)[0]
        mzval_.append(zval_[:, tloc].mean(1))
        mnval_.append(nval_[:, tloc].mean(1))
        mrval_.append(rval_[:, tloc].mean(1))
    mzval_ = np.vstack(mzval_).T
    mnval_ = np.vstack(mnval_).T
    mrval_ = np.vstack(mrval_).T

    # scikit hierarchical clustering
    if cluster_basis=='zscore':
        D = np.copy(mzval_[:, :])
    elif cluster_basis=='norm':
        D = np.copy(mnval_[:, :])
    elif cluster_basis=='raw':
        D = np.copy(mrval_[:, :])
    y = sch.linkage(D, method=link_method)
    z = sch.dendrogram(y, orientation='right', no_plot=True,
                       color_threshold=color_thresh * np.max(y[:, 2]))
    index = z['leaves']
    D = D[index, :]
    resulting_clusters = sch.fcluster(y, color_thresh * np.max(y[:, 2]),
                                      criterion='distance')

    # find if rhythmic, get amplitude
    rhythmic = []
    for di in ds_.index:
        if di in rhythmicity_data.index:
            try:
                ampt = np.float(rhythmicity_data['FoldC'][di])
            except ValueError:
                amps = rhythmicity_data['FoldC'][di].split(',')
                ampt = np.mean(np.array(amps, dtype='float'))
            rhythmic+=[1]
        else:
            rhythmic+=[0]

    # outputs
    clustering_dict= {
                    'labels': lab_,
                    'raw_expression': mrval_,
                    'z_expression': mzval_, 
                    'normed_expression': mnval_, 
                    'D':D, 
                    'y':y, 
                    'name':name,
                    'rhythmic':rhythmic, 
                    'clusters':resulting_clusters
                    }
    return clustering_dict

# make a plot
def cluster_plot(cluster_plot_info, ax=None, aax=None, bx=None):
    """
    Utility function for plotting the clusters that result and corresponding 
    info.
    """
    D= cluster_plot_info['D']
    y= cluster_plot_info['y']
    name =cluster_plot_info['name']
    rhythmic =cluster_plot_info['rhythmic']
    
    if ax is None:
        plt.figure(figsize=(4.0, 2.6))
        gs = gridspec.GridSpec(1, 3, width_ratios=(0.05, 1, 0.2))
        ax = plt.subplot(gs[:, 1])
        aax = plt.subplot(gs[:, 0])
        bx = plt.subplot(gs[:, 2])

    cbar = ax.pcolormesh(D[:, :], cmap='RdBu_r', vmax=3, vmin=-3)
    cbar.set_zorder(-20)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel('Time (h)')
    ax.set_xticks([0.5, 6.5, 12.5, 18.5])
    ax.set_xticklabels([0, 24, 48, 72])
    ax.set_ylabel(name + ' Transcript Expression (RPKM, z-Score)')
    ax.invert_yaxis()
    ax.set_rasterization_zorder(-10)
    #ax.axhline(500, color='red')

    cbar2 = aax.pcolormesh(np.array([rhythmic]).T, cmap='gray_r', vmax=0.1)
    cbar2.set_zorder(-20)
    aax.set_yticks([])
    aax.set_yticklabels([])
    aax.set_xticklabels([])
    aax.set_xticks([])
    #aax.axhline(500, color='red')
    aax.invert_yaxis()
    aax.set_rasterization_zorder(-10)

    
    sch.dendrogram(y, orientation='right', ax=bx,
                       color_threshold=color_thresh * np.max(y[:, 2]))
    bx.set_yticklabels([])
    bx.invert_yaxis()
    bx.axis('off')

    return cbar, cbar2

def plot_save_profile(cluster_name, **kwargs):
    fig = plt.figure(figsize=(3.5,2.5))
    gs = gridspec.GridSpec(1,2,width_ratios=(4,1))
    ax = plt.subplot(gs[0,0])
    cluster = cluster_expressions[cluster_name]
    dd_data = cluster['DD_norm']
    dd_max = np.percentile(dd_data, 90, 0)
    dd_min = np.percentile(dd_data, 10, 0)
    ld_data = cluster['LD_norm']
    ld_max = np.percentile(ld_data, 90, 0)
    ld_min = np.percentile(ld_data, 10, 0)
    N = len(ld_data[:,0])

    dd_raw = cluster['DD_raw']
    ld_raw = cluster['LD_raw']

    ax.fill_between(4*np.arange(len(dd_data[0,:])), dd_max,
                    dd_min, color='#708090', alpha=0.3)
    ax.fill_between(4*np.arange(len(ld_data[0,:])), ld_max,
                    ld_min, color="#ABDDDE", alpha=0.4)
    ax.plot(4*np.arange(len(dd_data[0,:])), dd_data.mean(0), ls='-', 
        label=f'{cluster_name} DD, n={N}', c='#708090', **kwargs)
    ax.plot(4*np.arange(len(ld_data[0,:])), ld_data.mean(0), ls='-', 
        label=f'{cluster_name} LD, n={N}', c="#ABDDDE", **kwargs)
    ax.set_xticks([0,24,48,72])
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Expression')
    ax.legend()

    bx = plt.subplot(gs[0,1])
    sns.violinplot(x=[0]*N, y=np.log10(dd_raw.mean(1)+1E-15),
                   color='#708090', ax=bx)
    sns.violinplot(x=[1]*N, y=np.log10(ld_raw.mean(1)+1E-15),
                   color="#ABDDDE", ax=bx)
    plt.setp(bx.collections, alpha=0.4)
    bx.set_ylim([-1,5])
    bx.set_ylabel('Log$_{10}$(Mean Expression)')

    plt.tight_layout(**plo.layout_pad)
    fig.savefig('results/figure_1/hierarchical/'+cluster_name+'.pdf')
    plt.close(fig)


#
# Process data
#

# load data
# pull into pandas dataframe
path = 'data/figures/hierarchical/'

# pull in the expression data
ds_names = ['DD', 'LD']
ds_DD = pd.read_csv(path + 'PcAS_WTDD input.txt', delimiter='\t', index_col=0)
ds_LD = pd.read_csv(path + 'PcAS_WTLD input.txt', delimiter='\t', index_col=0)
assert all(ds_LD.index==ds_DD.index), "Gene names differ."

# pull in the results of the rhythmicity tests
r_DD = pd.read_csv(path + 'Results 20-28hWTDD_Cycling wFoldChange.txt',
                   delimiter='\t', index_col=0)
r_LD = pd.read_csv(path + 'Results 20-28hWTLD_Cycling wFoldChange.txt',
                   delimiter='\t', index_col=0)


# parameters for clustering
color_thresh = 0.85  # threshold for separate clusters: default 0.7
link_method = 'complete'  # see scipy hierarchical clustering for details

# process the zDD and LD data to take a look
dd_dict =cluster_expression_data(ds_DD, 'DD', r_DD, color_thresh=color_thresh)
ld_dict =cluster_expression_data(ds_LD, 'LD', r_LD, color_thresh=color_thresh)
dd_clusters = dd_dict['clusters']
ld_clusters = ld_dict['clusters']
cluster_id_array = np.vstack([dd_clusters, ld_clusters]).T

# find most common combinations
dd_clusters = cluster_id_array[:, 0]
ld_clusters = cluster_id_array[:, 1]

# just look for most common combos
ll = [str(l) for l in cluster_id_array.tolist()]
all_combos = Counter(ll)
cluster_ids = np.sort(list(all_combos.keys()))

# perform fisher's exact test
dd_counter = Counter(dd_clusters)
ld_counter = Counter(ld_clusters)

fisher_dict = {}
for ddi in np.sort(list(dd_counter.keys())):
    for ldi in np.sort(list(ld_counter.keys())):
        # get potential cluster IDs
        cid = f'[{ddi}, {ldi}]'
        if cid in cluster_ids:
            inDD = dd_counter[ddi]
            inLD = ld_counter[ldi]
            inBOTH = all_combos[cid]
            inNEITHER = len(dd_clusters) - inDD + inBOTH - inLD
            fisher_mat = np.array([[inBOTH, inDD-inBOTH], 
                                   [inLD-inBOTH, inNEITHER]])

            # bonferroni p
            bp = len(cluster_ids)
            fisher_dict[cid] = stats.fisher_exact(fisher_mat, 
                                    alternative='greater')[1]*bp

result_dict = {}
for key in cluster_ids:
    if fisher_dict[key] < 0.05:
        result_dict[key] = all_combos[key]

# Plot what the resulting clusters look like
plo.PlotOptions(ticks='in')
plt.figure(figsize=(2.0, 2.6))
gs = gridspec.GridSpec(1, 3, width_ratios=(0.05, 1, 0.2))
ax = plt.subplot(gs[:, 1])
aax = plt.subplot(gs[:, 0])
bx = plt.subplot(gs[:, 2])
cbar, cbar2 = cluster_plot(dd_dict, ax=ax, aax=aax, bx=bx)
plt.tight_layout(**plo.layout_pad)


plt.figure(figsize=(2.0, 2.6))
gs = gridspec.GridSpec(1, 3, width_ratios=(0.05, 1, 0.2))
ax = plt.subplot(gs[:, 1])
aax = plt.subplot(gs[:, 0])
bx = plt.subplot(gs[:, 2])
cbar, cbar2 = cluster_plot(ld_dict, ax=ax, aax=aax, bx=bx)
plt.tight_layout(**plo.layout_pad)

# plt.figure(figsize=(3.0, 1.3))
# plt.colorbar(cbar)
# plt.tight_layout()

# compile relevant genes

# generate the resulting clusters and save them
cluster_names =['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
cluster_expressions = {}
categorized_idxs = [] # get the indexes that have been clustered
for idd, res in enumerate(list(result_dict.keys())):
    # result key
    resl = eval(res)

    # set up lsit outputs
    temp_list = []
    norm_dd = []
    norm_ld = []
    raw_dd = []
    raw_ld = []
    rhyth_dd = []
    rhyth_ld =[]
    for idx, cl in enumerate(cluster_id_array):
        if np.all(np.array(resl)==cl):
            gene_name = ds_DD.index[idx]
            categorized_idxs.append(idx)

            # making sure we are at the same place in the arrays
            assert gene_name==ds_LD.index[idx], "ensuring sorting failed"
            # add gene to list
            temp_list+=[gene_name]

            if gene_name in r_DD.index:
                rhyth_dd.append(1)
            else:
                rhyth_dd.append(0)
            if gene_name in r_LD.index:
                rhyth_ld.append(1)
            else:
                rhyth_ld.append(0)
            norm_dd.append(dd_dict['normed_expression'][idx])
            norm_ld.append(ld_dict['normed_expression'][idx])
            raw_dd.append(dd_dict['raw_expression'][idx])
            raw_ld.append(ld_dict['raw_expression'][idx])

    norm_dd = np.vstack(norm_dd)
    norm_ld = np.vstack(norm_ld)
    raw_dd = np.vstack(raw_dd)
    raw_ld = np.vstack(raw_ld)
    # assemble the expression for each cluster
    cluster_info = {'DDLD_sorting': resl, 
                    'DD_norm': norm_dd, 
                    'LD_norm': norm_ld,
                    'DD_raw' : raw_dd,
                    'LD_raw' : raw_ld,
                    't' : 4*np.arange(len(raw_dd[0,:])),
                    "name": cluster_names[idd],
                    'DD_rhyth' : rhyth_dd,
                    'LD_rhyth' : rhyth_ld
                    }
    cluster_expressions[cluster_names[idd]] = cluster_info

    np.savetxt('results/figure_1/hierarchical/'+cluster_names[idd]+'.csv', 
               np.array(temp_list), fmt="%s", delimiter=',')

# add an uncategorized cluster
uncat = []
# set up lsit outputs
temp_list = []
norm_dd = []
norm_ld = []
raw_dd = []
raw_ld = []
rhyth_dd = []
rhyth_ld =[]
# find which idxs are unclassified
for idx in np.arange(len(cluster_id_array)):
    if idx not in categorized_idxs:
        gene_name = ds_DD.index[idx]
        # making sure we are at the same place in the arrays
        assert gene_name==ds_LD.index[idx], "ensuring sorting failed"
        # add gene to list
        temp_list+=[gene_name]

        if gene_name in r_DD.index:
            rhyth_dd.append(1)
        else:
            rhyth_dd.append(0)
        if gene_name in r_LD.index:
            rhyth_ld.append(1)
        else:
            rhyth_ld.append(0)
        norm_dd.append(dd_dict['normed_expression'][idx])
        norm_ld.append(ld_dict['normed_expression'][idx])
        raw_dd.append(dd_dict['raw_expression'][idx])
        raw_ld.append(ld_dict['raw_expression'][idx])
norm_dd = np.vstack(norm_dd)
norm_ld = np.vstack(norm_ld)
raw_dd = np.vstack(raw_dd)
raw_ld = np.vstack(raw_ld)
# assemble the expression for each cluster
cluster_info = {'DDLD_sorting': resl, 
                'DD_norm': norm_dd, 
                'LD_norm': norm_ld,
                'DD_raw' : raw_dd,
                'LD_raw' : raw_ld,
                't' : 4*np.arange(len(raw_dd[0,:])),
                "name": 'Uncategorized',
                'DD_rhyth' : rhyth_dd,
                'LD_rhyth' : rhyth_ld
                }
cluster_expressions['Uncategorized'] = cluster_info
np.savetxt('results/figure_1/hierarchical/Uncategorized.csv', 
               np.array(temp_list), fmt="%s", delimiter=',')


# get means of each gene within each cluster
dd_means_by_cluster = []
ld_means_by_cluster = []
for cc in cluster_names:
    dd = cluster_expressions[cc]['DD_raw']
    ld = cluster_expressions[cc]['LD_raw']
    dd_means_by_cluster.append(np.mean(dd, 1))
    ld_means_by_cluster.append(np.mean(ld, 1))

# plot profiles and means
for name in cluster_expressions.keys():
    plot_save_profile(name)

# plot of a couple of clusters for example

# table names
# we need to modify the cluster_expressions dict into a pandas array
# include: cluster name, number of genes
def cluster_details(ce):
    """
    Takes a cluster expression dict and returns summary stats:
    cluster name, number of genes, mean expression level, fraction cycling,
    phase, waveform (done after fitting). Waveform and phase only done if
    >50% of cluster oscillates in LD.
    """
    # num genes
    num_genes = ce['DD_raw'].shape[0]

    # mean expression of each genes
    mes_ld = ce['LD_raw'].mean(1)
    mes_dd = ce['DD_raw'].mean(1)
    me_ld = mes_ld.mean()
    me_dd = mes_dd.mean()

    # # mean profiles
    # mp_ld = ce['LD_raw'].mean(0)
    # mp_dd = ce['DD_raw'].mean(0)

    # fraction cycling
    dd_cycling = np.mean(ce['DD_rhyth'])
    #dd_cloc = np.where(ce['DD_rhyth'])[0]
    #dd_fold = np.mean(np.max(ce['DD_raw'][dd_cloc],1)/np.mean(ce['DD_raw'][dd_cloc],1))
    ld_cycling = np.mean(ce['LD_rhyth'])
    #ld_cloc = np.where(ce['LD_rhyth'])[0]
    #ld_fold = np.mean(np.max(ce['LD_raw'][dd_cloc],1)/np.mean(ce['LD_raw'][dd_cloc],1))

    # phase, waveform (string) from fitting NORMED data
    if ld_cycling > 0.5:
        ld_shape = np.mean(ce['LD_norm'],0)
        period = 24
        t = ce['t']
        sinefit = pro.fit_sin(t, ld_shape, 24, 0.1)
        squarefit = pro.fit_square(t, ld_shape, 24, 0.1)
        sawfit = pro.fit_saw(t, ld_shape, 24, 0.1)
        sawfit2 = pro.fit_saw(t, ld_shape, 24, 0.1, reverse=True)

        # plot
        plt.figure()
        plt.plot(t, ld_shape, 'k', lw=2)
        plt.plot(t, sinefit['fitfunc'](t))
        plt.plot(t, squarefit['fitfunc'](t))
        plt.plot(t, sawfit['fitfunc'](t))
        plt.plot(t, sawfit2['fitfunc'](t))

        # compare fits
        fits = [sinefit, squarefit, sawfit, sawfit2]
        r2s = [fit['pseudoR2'] for fit in fits]
        
        fit = fits[np.argmax(r2s)]
        waveform = fit['name']
        r2 = fit['pseudoR2']
        phase = fit['phase']*24/(2*np.pi) %24.
        amp = fit['amp']
    else:
        phase='Non-Oscillatory'
        waveform='Non-Oscillatory'
        r2 = 0
        amp = 0

    return [ce['name'], num_genes, me_dd, dd_cycling, me_ld, ld_cycling, phase, amp, waveform, r2]

titles = ['Cluster Name', 'Gene Count', 'Mean DD Expression (RPKM)',
          'Fraction Oscillatory DD', 'Mean LD Expression (RPKM)', 
          'Fraction Oscillatory LD', 'LD Peak Phase (0 to 24h)', 
          'LD Mean Amplitude', 'Oscillatory Waveform', 'R2']

results = []
for cn in cluster_expressions.keys():
    print(cn)
    ce = cluster_expressions[cn]
    results.append(cluster_details(ce))

table = pd.DataFrame(results, columns=titles)
table.to_csv('results/figure_1/cluster_results.csv')
