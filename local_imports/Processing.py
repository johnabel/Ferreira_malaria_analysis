# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2017

@author: john abel
Module set up to perform analysis for Yongli Shan.
"""

from concurrent import futures
import itertools
import numpy  as np
from scipy import signal, stats, optimize
from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve
from sklearn import decomposition, cluster, mixture
import matplotlib.pyplot as plt
import pywt

from . import DecayingSinusoidRNA as dsin
from . import PlotOptions as plo
from . import Utilities as uts

import collections



def parallel_MIC(data, num_cpus=20):
    """
    Parallel calculation of MIC for an array of trajectories.

    Parameters
    ----------
    data
    num_cpus : int (default=20)
        Number of CPUs to allocate to the parallel process.
    """
    dlen = len(data)

    # get the index combinations
    inds = list(itertools.combinations_with_replacement(
                range(dlen),2))

    data_inputs = []
    # set up the input file to the parallelization
    for ind in inds:
        # input looks like: index1, index2, data1, data2
        data_inputs.append([ind[0],ind[1], data[ind[0]],data[ind[1]]])

    with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        result = executor.map(single_MIC, data_inputs)

    # process the result object that is output
    mic = np.zeros([dlen,dlen])
    for r in result:
        mic[r[0],r[1]] =r[2]
        mic[r[1],r[0]] =r[2]

    np.fill_diagonal(mic,0)
    return mic


def full_MIC(data):
    """
    Serial calculation of MIC for an array of trajectories.

    Parameters
    ----------
    use_data : str (default='raw')
        Key of the Network.data dictionary from which the data is pulled.
    """
    miner = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    c1 = range(len(data))
    c2 = range(len(data))

    mic = np.zeros([len(data), len(data)])
    for i in c1:
        x1 = data[i]
        for j in c2:
            if i<=j:
                x2 = data[j]
                miner.compute_score(x1,x2)
                mic[i,j] = miner.mic()
            else:
                mic[i,j] = mic[j,i]
        if i%1000==0:
            print(i)

    return mic


def MIC(d1, d2):
    """
    Serial calculation of MIC for an array of trajectories.

    Parameters
    ----------
    use_data : str (default='raw')
        Key of the Network.data dictionary from which the data is pulled.
    """
    miner = mp.MINE(alpha=0.6, c=15, est="mic_approx")

    miner.compute_score(d1,d2)


    return mic

def single_MIC(data_t1t2):
    """
    MIC calculation for a single pair of trajectories. Eliminates NaNs.
    """
    t1, t2, d1, d2 = data_t1t2
    d1nan = np.isnan(d1)
    d2nan = np.isnan(d2)
    d1 = d1[~d1nan & ~d2nan]
    d2 = d2[~d1nan & ~d2nan]
    miner = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    miner.compute_score(d1, d2)
    return t1,t2,miner.mic()

def periodogram(x, y, period_low=1, period_high=60, res=200):
    """ calculate the periodogram at the specified frequencies, return
    periods, pgram """

    periods = np.linspace(period_low, period_high, res)
    # periods = np.logspace(np.log10(period_low), np.log10(period_high),
    #                       res)
    freqs = 2*np.pi/periods
    pgram = signal.lombscargle(x, y, freqs, precenter=True)
    var = np.var(y)

    #take the normalized power
    pgram_norm = pgram *2 / (len(x) * var)

    return periods, pgram, pgram_norm

def LS_pgram_RNAseq(times, d1, per_low=20, per_high=30, threshold=0.6, res=401):
    """Calculates a LS periodogram for each data sequence,
    and returns the periodogram value. If the largest significant
    peak is in the circadian range as specified by the args, it is
    rhythmic. The threshold for rhythmicity must be predefined, usually
    by bootstrapping."""


    # remove nans
    t1 = np.copy(times[~np.isnan(d1)])
    d1 = np.copy(d1[~np.isnan(d1)])

    if len(t1)>0:
        pers, pgram, pgram_norm = periodogram(t1, d1, period_low=per_low,
                        period_high=per_high, res=res)

        peak = np.max(pgram)
        rhythmic = peak > threshold
        period = pers[np.argmax(pgram)]
    else:
        peak = 0
        rhythmic = False
        period = 0

    return peak, rhythmic, period
'''
def identify_LS_threshold(times, replicates, alpha=0.05, variance=1,
                          bootstrap=5E4, per_low=20, per_high=30, res=1001):
    """Finds the threshold for LS Periodogram given a known set of times
    and variances. Presumes each individual is z-scored independently."""

    zscore_data = []
    for replicate in range(replicates):
        rep_data = np.random.normal(size=[int(bootstrap), len(times)])
        zscore_data.append(rep_data)
    zscore_data = np.array(zscore_data)
    ls_null_data = np.mean(zscore_data, axis=0)

    pgrams = []
    for data in ls_null_data:
        pers, pgram, pgram_norm = periodogram(times, data, period_low=per_low,
                    period_high=per_high, res=res)
        pgrams.append(np.max(pgram))

    values, base = np.histogram(pgrams, bins=500)
    cumulative = (bootstrap - np.cumsum(values))/bootstrap
    threshold = base[np.where(cumulative<alpha)[0][0]]
    return threshold
'''

def identify_LS_threshold2(times, replicates, alpha=0.05,
                          bootstrap=5E4, per_low=20, per_high=30, res=401):
    """Finds the threshold for LS Periodogram given a known set of times
    and assuming the data are all Gaussian noise. The samples should be
    z-scored before being compared to this statistic. """


    rep_data = np.random.normal(size=[int(bootstrap), 
                                int(len(times)*replicates)])

    rep_data = stats.zscore(rep_data, axis=1)
    zscore_data = np.array([rep_data[:,idx*len(times):(idx+1)*len(times)] for idx in range(replicates)])
    ls_null_data = np.mean(zscore_data, axis=0)

    pgrams = []
    for data in ls_null_data:
        pers, pgram, pgram_norm = periodogram(times, data, period_low=per_low,
                    period_high=per_high, res=res)
        pgrams.append(np.max(pgram))

    values, base = np.histogram(pgrams, bins=500)
    cumulative = (bootstrap - np.cumsum(values))/bootstrap
    threshold = base[np.where(cumulative<alpha)[0][0]]
    return threshold

def pca(data, n_components=None):
    """ returns a sklearn.decomposition.pca.PCA object fit to the data """
    pca = decomposition.pca.PCA(n_components=n_components)
    pca.fit(data)
    return pca

def plot_heatmap(pcs_data, pc1=0, pc2=1, return_fig=False, return_img=True):

    fig = plt.figure(figsize=(6,4))
    heatmap, xedges, yedges = np.histogram2d(
            pcs_data[:,pc1], pcs_data[:,pc2], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    ax = plt.subplot()
    plt.imshow(heatmap.T, extent=extent, origin=0, aspect='auto')
    ax.set_xlabel('PC'+str(pc1+1)); ax.set_ylabel('PC'+str(pc2+1))

def find_common_genes(set1, set2, dl):
    """
    Find correspondence between set2 data and set1 data. Note that 
    we assume that dl is the overall ordered list of transcripts that is 
    reduced to become set1 and set2. Also, there might be a faster way hare.

    Arguments:
    ----------
    set1 : list
        Binary rhythmic (1) or arrhythmic (0) for dataset1.
    set2 : list
        Binary rhythmic (1) or arrhythmic (0) for dataset2.
    dl : list
        List of all transcripts.
    """
    # things to collect here are
    agree_rhythmic_locs = [] # locations of the data in the sets that agree
    for idx, gene in enumerate(dl):
        if np.logical_and(set1[idx]==1, set2[idx]==1):
            agree_rhythmic_locs.append(idx)
    return np.array(agree_rhythmic_locs)

def find_common_genenames(set1, set2):
    """
    Find correspondence between set2 data and set1 gene names.

    Arguments:
    ----------
    set1 : list
        List of genes in dataset1.
    set2 : list
        List of genes in dataset2.
    """
    assert len(set1)<=len(set2), 'Dataset 1 must be at most as long as dataset 2.'
    assert len(np.unique(set1))/len(set1)==1., 'Dataset 1 must consist of unique entries.'
    assert len(np.unique(set2))/len(set2)==1., 'Dataset 2 must consist of unique entries.'
    
    # things to collect here are
    set1_idxs = []
    set2_idxs = []
    for idx, gene in enumerate(set1):
        if set1[idx]==set2[idx]:
            # perf case, 
            set1_idxs.append(idx)
            set2_idxs.append(idx)
        elif np.isin(gene, set2):
            idx2 = np.where(np.array(set2)==gene)[0]
            if len(idx2)==1:
                set1_idxs.append(idx)
                set2_idxs.append(idx2[0])
        else: 
            pass
    return set1_idxs, set2_idxs


def fit_sin(tt, yy, guess_per, per_bound = 1):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    tt, yy = _pop_nans(tt, yy)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi/guess_per, 0, guess_offset])
    
    w_min = 2.*np.pi/(guess_per+per_bound)
    w_max = 2.*np.pi/(guess_per-per_bound)

    def sinfunc(t, A, w, p, c):  return A * np.cos(w*tt - p) + c
    bounds = ([0, w_min, -2*np.pi, -np.inf], [np.inf, w_max, 2*np.pi, np.inf])
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess, bounds=bounds,
                                    max_nfev=10E3, xtol=1E-4, ftol=1E-4)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.cos(w*t - p) + c

    SSresidual = np.sum((yy-fitfunc(tt))**2)
    SStot = np.sum((yy-np.mean(yy))**2)
    pseudoR2 = 1-SSresidual/SStot
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov), "pseudoR2":pseudoR2, "name": 'sine'}

def fit_square(tt, yy, guess_per, per_bound = 1):
    '''Fit square wave to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi/guess_per, 0, guess_offset])
    
    w_min = 2.*np.pi/(guess_per+per_bound)
    w_max = 2.*np.pi/(guess_per-per_bound)

    def squarefunc(t, A, w, p, c):  return A * signal.square(w*tt - p) + c
    bounds = ([0, w_min, -2*np.pi, -np.inf], [np.inf, w_max, 2*np.pi, np.inf])
    popt, pcov = optimize.curve_fit(squarefunc, tt, yy, p0=guess, bounds=bounds)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * signal.square(w*t - p) + c

    SSresidual = np.sum((yy-fitfunc(tt))**2)
    SStot = np.sum((yy-np.mean(yy))**2)
    pseudoR2 = 1-SSresidual/SStot
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov), "pseudoR2":pseudoR2, "name": 'square'}

def fit_saw(tt, yy, guess_per, per_bound = 1, reverse=False):
    '''Fit square wave to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    if reverse:
        flip=-1
        name='sawtooth_r'
    else:
        flip=1
        name='sawtooth'

    tt = np.array(tt)
    yy = np.array(yy)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi/guess_per, 0, guess_offset])
    
    w_min = 2.*np.pi/(guess_per+per_bound)
    w_max = 2.*np.pi/(guess_per-per_bound)

    def sawfunc(t, A, w, p, c):  return A * signal.sawtooth(flip*w*tt - p) + c
    bounds = ([0, w_min, -2*np.pi, -np.inf], [np.inf, w_max, 2*np.pi, np.inf])
    popt, pcov = optimize.curve_fit(sawfunc, tt, yy, p0=guess, bounds=bounds)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * signal.sawtooth(flip*w*t - p) + c

    SSresidual = np.sum((yy-fitfunc(tt))**2)
    SStot = np.sum((yy-np.mean(yy))**2)
    pseudoR2 = 1-SSresidual/SStot
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov), "pseudoR2":pseudoR2, "name": name}

def detrend(x, y, est_period=24., ret="detrended", a=0.05):
    """ Detrend the data using a hodrick-prescott filter. If ret ==
    "mean", return the detrended mean of the oscillation. Estimated
    period and 'a' parameter are responsible for determining the optimum
    smoothing parameter """

    x = np.asarray(x)
    y = np.asarray(y)

    # yt, index = timeseries_boundary(y, opt_b='mir', bdetrend=False)

    # As recommended by Ravn, Uhlig 2004, a calculated empirically
    num_periods = (x.max() - x.min())/est_period
    points_per_period = len(x)/num_periods
    w = a*points_per_period**4


    y_mean = hpfilter(y, w)
    y_detrended = y - y_mean

    if ret == "detrended": return x, y_detrended
    elif ret == "mean": return x, y_mean
    elif ret == "both": return x, y_detrended, y_mean


def hpfilter(X, lamb):
    """ Code to implement a Hodrick-Prescott with smoothing parameter
    lambda. Code taken from statsmodels python package (easier than
    importing/installing, https://github.com/statsmodels/statsmodels """

    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = speye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = dia_matrix((data, offsets), shape=(nobs-2,nobs))

    trend = spsolve(I+lamb*K.T.dot(K), X, use_umfpack=True)
    return trend

def _pop_nans(x, y):
    """ Remove nans from incoming dataset """
    xnan = np.isnan(x)
    ynan = np.isnan(y)
    return x[~xnan & ~ynan], y[~xnan & ~ynan]


def load_invivo_data(path, binning=None, sample_days=np.array(
        [4, 5, 6, 7, 8, 11, 12, 13, 14]), start_day=[9, 25], infection_day=[9, 26], sample_zt=2):
    """
    loads the wheel running rhythms from a csv located at path
    if binning is int, bins into that many minute units

    infection_day is when the infection occurs, start_day is when the running starts.
    should be a date in terms [month, day].

    Note that month is included only as a safety check on the time alignment as
    it should not differ.
    """
    # load the data and assemble parts
    data = np.loadtxt(path, delimiter=',', dtype=str)
    counts = data[4:, 3]
    recordings = [loc for loc in range(len(counts)) if counts[loc].isdigit()]
    days = data[4:, 0].astype(float) - data[4, 0].astype(float)
    # (because starts at day nonzero)
    hours = data[4:, 1].astype(float)
    mins = data[4:, 2].astype(float)
    times = np.asarray([days[i] * 24 + (hours[i] + mins[i] / 60)
                        for i in range(len(days))])

    # get into format times, countsT
    ret_times = times[recordings]
    ret_counts = counts[recordings].astype(float)

    # adjust the times such that infection day is day 0

    times_offset = 24 * (infection_day[1] - start_day[1])
    if infection_day[0] - start_day[0] != 0:
        print("WARNING: difference in month, correct in input")

    ret_times = ret_times - times_offset
    # if binning
    if isinstance(binning, int):
        new_times = ret_times[::binning]
        bins = new_times
        digitized = np.digitize(ret_times, bins)
        new_counts = np.array([ret_counts[digitized == i].sum()
                               for i in range(1, len(bins))])
        # re-assign
        ret_counts = new_counts
        ret_times = new_times[:-1]

    # collect where samples were taken
    sample_times = []
    for sample_day in sample_days:
        # first sample is taken at day 4
        sample_times.append((sample_day) * 24 + 6 + sample_zt)

    return {'times': ret_times,
            'counts': ret_counts,
            'path': path,
            'sample_times': np.array(sample_times)
            }


def find_cwt0_region(region, x, mhw):
    """ finds the zero-cross of the mexican hat wavelet, as suggested in
    Leise 2013. """
    specific_region = mhw[region[0]:region[1]]
    specific_x_locs = np.arange(len(specific_region)) + region[0]
    specific_times = x[specific_x_locs]

    # find peaks
    zeros_loc = np.where(np.diff(np.sign(specific_region)))[0]
    # spline interpolate for simplicity
    spl = uts.UnivariateSpline(specific_times, specific_region, k=3, s=0)

    onset_time = spl.roots()
    if len(onset_time) > 0:
        return onset_time[0]


def cwt_onset(data_dict, cc_data=None, N=6):
    """ I'm fitting the
    first N onsets, then drawing an average line through them"""
    x = data_dict['times']
    y = data_dict['counts']
    bin_param = 15
    widths = np.arange(3 * (60 / bin_param),
                       9 * (60 / bin_param),
                       0.05 * (60 / bin_param))

    # take the mexican hat waveform
    cwtmatr, freqs = pywt.cwt(y, widths, 'mexh')
    periods = 1 / freqs / (60 / bin_param)

    inst_per_loc = np.argmax(np.abs(cwtmatr.T), 1)
    inst_ampl = np.asarray([cwtmatr.T[idx, loc]
                            for idx, loc in enumerate(inst_per_loc)])

    # identify regions of increasing activity
    maxs = signal.argrelextrema(inst_ampl, np.greater, order=40)[0]
    mins = signal.argrelextrema(inst_ampl, np.less, order=40)[0]

    # find the 0-crosses here
    inc_regions = []
    for mini in mins:
        try:
            maxi = np.min(maxs[np.where(maxs > mini)])
            inc_regions.append([mini, maxi])
        except ValueError:
            # if there is no following max
            pass

    # get the onset times
    onsets = []
    for region in inc_regions[:]:
        onset = find_cwt0_region(region, x, inst_ampl)
        if onset is not None:
            onsets.append(onset)
    onsets = np.asarray(onsets)

    # now find period and assign phases
    onsets_used = N
    onsets = onsets[:N]
    onset_idxs = np.arange(len(onsets[:(onsets_used)]))

    # generate slices of 48h incrementing up by 24h
    # center the first onset at 25h
    # attach nearby onsets as if we are constructing an actogram
    slices = [[xx, xx + 48]
              for xx in np.arange(onsets[0] - 25, np.max(x) + 48, 24)]
    slice_onsets = []
    slices_used = []
    for sidx, slicei in enumerate(slices):
        # append onset times relative to slice start
        # these are all onsets in the slice
        sos = onsets[np.logical_and(
            onsets < slicei[1], onsets > slicei[0])] - slicei[0]
        if sidx == 0:
            slice_onset = sos[0]  # always the first onset, should be 25
            slice_onsets.append(slice_onset)
            slices_used.append(sidx)
        elif len(sos) > 0:
            so_diffs = np.sqrt((sos - slice_onset)**2)
            # if one is within 10h, append it
            if np.min(so_diffs) < 10:
                slice_onset = sos[np.argmin(so_diffs)]
                slice_onsets.append(slice_onset)
                slices_used.append(sidx)
        else:
            pass  # if there is no onset

    #plt.plot(slices_used, slice_onsets, 'ko')

    slope, x0, _, _, _ = stats.linregress(slices_used, slice_onsets)
    period = 24 + slope
    phases = (2 * np.pi * (x - (x0 - 25 + onsets[0])) / period) % (2 * np.pi)

    #plt.plot(x, phases)
    #plt.plot(onsets,[0]*len(onsets), 'ko')

    # get the sample locations
    sample_times = data_dict['sample_times']
    sample_locs = []
    for st in sample_times:
        sample_locs.append(np.argmin(np.abs(x - st)))
    sample_locs = np.array(sample_locs)

    phases_of_samples = (
        (sample_times - (x0 - 25 + onsets[0])) * 2 * np.pi / period) % (2 * np.pi)
    sample_locs = np.array(sample_locs)

    data_dict['x'] = x
    data_dict['y'] = y
    data_dict['onsets'] = np.array(onsets)
    data_dict['dwt'] = cwtmatr
    data_dict['period'] = period
    data_dict['phases'] = phases
    data_dict['cycle_lengths'] = np.diff(onsets[:(onsets_used)])
    data_dict['regions'] = np.array(inc_regions)
    data_dict['sample_inds'] = sample_locs  # only works if time covers it
    data_dict['sample_phases'] = phases_of_samples
    data_dict['onsets_fit'] = x[signal.argrelextrema(phases, np.less)]

    if cc_data is not None:
        data_dict['cc'] = cc_data
    return data_dict

