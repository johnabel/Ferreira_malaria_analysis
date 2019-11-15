# -*- coding: utf-8 -*-
"""
Python 3.7
1 Feb 2019

The point of this file is to find:
- the time-difference between onsets each day to approximate
  the circadian period

ZT 0 IS 6AM!!!!!

There are two options on how to do this.
Option 1 (implemented in v0 of this code):
    Reassign phase based on mouse activity phase. Test if there are rhythms 
    with the periodicity of the mouse.

Option 2 (implemented in the archived version of this cose):
    Set the time at the first onset to 0. Set the "sample_phase" to be
    all the times relative to that first onset. Use LSP to get period of that
    dataset. (This works as well and shows that the period is lengthened for
    FB.)

@author: john abel
"""

import numpy as np
import scipy as sp
from scipy import signal, stats
from scipy.interpolate import UnivariateSpline
import pywt
import scikit_posthocs as sph

import matplotlib.pyplot as plt
from matplotlib import gridspec

from local_imports import PlotOptions as plo
from local_imports import Processing as pro
from local_imports import Utilities as util

#
#
#   |D |E |F |I |N |I |T |I (O |N |S
#


# next: get phase from wheel running data (use cristina code)
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
    spl = UnivariateSpline(specific_times, specific_region, k=3, s=0)

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
            # if one is within 6h, append it
            if np.min(so_diffs) < 6:
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


#
#
#
#

#
# Part 1: collect the extended cell cycle data and wheel rhythms
# FBXL3 cell cycle dynamics-2.xlsx, sheet "Sheet1"
# first: assemble the cell cycle data
day = np.array([4, 5, 6, 7, 8, 11, 12, 13, 14])
f157 = np.array([0.331, 0.411, 0.346, 0.397,
                 0.232, 0.788, 0.513, 0.496, 0.577])
f158 = np.array([0.459, 0.493, 0.395, 0.406,
                 0.255, 0.798, 0.581, 0.325, 0.369])
f160 = np.array([0.573, 0.467, 0.374, 0.344,
                 0.198, 0.718, 0.565, 0.545, 0.569])
f161 = np.array([0.619, 0.462, 0.341, 0.437,
                 0.350, 0.743, 0.688, 0.540, 0.415])
f168 = np.array([0.752, 0.400, 0.366, 0.391,
                 0.402, 0.714, 0.660, 0.357, 0.453])
wt1cc = np.array([0.732, 0.574, 0.614, 0.604,
                  0.657, 0.294, 0.220, 0.284, 0.336])
wt2cc = np.array([0.810, 0.657, 0.636, 0.341,
                  0.602, 0.374, 0.330, 0.651, 0.470])
wt3cc = np.array([0.807, 0.588, 0.648, 0.629,
                  0.619, 0.373, 0.500, 0.302, 0.433])
wt4cc = np.array([0.724, 0.680, 0.647, 0.586,
                  0.564, 0.381, 0.514, 0.368, 0.605])


# collect the info
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/404c_F157_Counts.csv'
F157 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                        infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/404c_F168_Counts.csv'
F168 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                        infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/404c_F158_Counts.csv'
F158 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                        infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/404c_F160_Counts.csv'
F160 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                        infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/404c_F161_Counts.csv'
F161 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                        infection_day=[9, 26], sample_zt=2)

N1 = 6
F157 = cwt_onset(F157, f157, N=N1)
F158 = cwt_onset(F158, f158, N=N1)
F160 = cwt_onset(F160, f160, N=N1)
F161 = cwt_onset(F161, f161, N=N1)
F168 = cwt_onset(F168, f168, N=N1)


# collect the info
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/WT1_male_Counts.csv'
WT1 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                       infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/WT2_fem_Counts.csv'
WT2 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                       infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/WT3_fem_Counts.csv'
WT3 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                       infection_day=[9, 26], sample_zt=2)
path = 'data/figures/wheel_ratios/Exp24_DYNAMICS/WT4_fem_Counts.csv'
WT4 = load_invivo_data(path, binning=15, sample_days=day, start_day=[9, 25],
                       infection_day=[9, 26], sample_zt=2)


WT1 = cwt_onset(WT1, wt1cc, N=N1)
WT2 = cwt_onset(WT2, wt2cc, N=N1)
WT3 = cwt_onset(WT3, wt3cc, N=N1)
WT4 = cwt_onset(WT4, wt4cc, N=N1)


# get the cycle lengths from these FB and WT mice
NS = 5  # only using first five so that we only have the part we know for sure
collected_fb_phases1 = []
collected_fb_phases1.append(F157['sample_phases'][:NS])
collected_fb_phases1.append(F158['sample_phases'][:NS])
collected_fb_phases1.append(F160['sample_phases'][:NS])
collected_fb_phases1.append(F161['sample_phases'][:NS])
collected_fb_phases1.append(F168['sample_phases'][:NS])
collected_wt_phases1 = []
collected_wt_phases1.append(WT1['sample_phases'][:NS])
collected_wt_phases1.append(WT2['sample_phases'][:NS])
collected_wt_phases1.append(WT3['sample_phases'][:NS])
collected_wt_phases1.append(WT4['sample_phases'][:NS])

collected_fb_periods1 = [F157['period'], F158['period'], F160['period'],
                         F161['period'], F168['period']]
collected_wt_periods1 = [WT1['period'], WT2['period'],
                         WT3['period'], WT4['period']]

# and their periods too
collected_fb_cc1 = []
collected_fb_cc1.append(F157['cc'][:NS])
collected_fb_cc1.append(F158['cc'][:NS])
collected_fb_cc1.append(F160['cc'][:NS])
collected_fb_cc1.append(F161['cc'][:NS])
collected_fb_cc1.append(F168['cc'][:NS])
collected_wt_cc1 = []
collected_wt_cc1.append(WT1['cc'][:NS])
collected_wt_cc1.append(WT2['cc'][:NS])
collected_wt_cc1.append(WT3['cc'][:NS])
collected_wt_cc1.append(WT4['cc'][:NS])


# Collect WT and FB #1
collected_fb_cc1 = np.hstack(collected_fb_cc1)
collected_fb_phases1 = np.hstack(collected_fb_phases1)
collected_wt_cc1 = np.hstack(collected_wt_cc1)
collected_wt_phases1 = np.hstack(collected_wt_phases1)


print("Finished long-duration experiments. Printing names of mice that should be excluded.")


#
#  PART 2: Collect the two days of cell cycle data from FB mice
#   FBXL3 cell cycle dynamics.xlsx, sheet "FB 24h resolution ratios"

FB_rep_times = np.array([0, 0, 4, 4, 8, 8, 12, 12, 16, 16,
                         20, 20, 24, 24, 28, 28, 32, 32, 36, 36,
                         40, 40, 44, 44, 48, 48, 48, 52, 52, 56, 56,
                         60, 60, 64, 64, 68, 68, 72, 72])
FB_rep_cc = np.array([0.484, 0.429,
                      0.389, 0.483,
                      0.362, 0.348,
                      0.219, 0.359,
                      0.281, 0.299,
                      0.308, 0.318,
                      0.341, 0.354,
                      0.417, 0.399,
                      0.454, 0.463,
                      0.488, 0.537,
                      0.443, 0.433,
                      0.344, 0.331,
                      0.248, 0.242, 0.221,
                      0.390, 0.402,
                      0.537, 0.548,
                      0.462, 0.476,
                      0.336, 0.353,
                      0.293, 0.282,
                      0.355, 0.316])
path_root = 'data/figures/wheel_ratios/FB24h_resolution_ratios/'
# ignore where missing
files = ['FBXL3 Pc - F01_Counts.csv', 'FBXL3 Pc - F02_Counts.csv',
         'FBXL3 Pc - F03_Counts.csv', 'FBXL3 Pc - M04_Counts.csv',
         'FBXL3 Pc - M05_Counts.csv', 'FBXL3 Pc - M06_Counts.csv',
         'FBXL3 Pc - M07_Counts.csv', 'FBXL3 Pc - M08_Counts.csv',
         'FBXL3 Pc - M09_Counts.csv', 'FBXL3 Pc - M10_Counts.csv',
         'FBXL3 Pc - M11_Counts.csv', 'FBXL3 Pc - M12_Counts.csv',
         'FBXL3 Pc - F13_Counts.csv', 'FBXL3 Pc - F14_Counts.csv',
         'FBXL3 Pc - F15_Counts.csv', 'FBXL3 Pc - M16_Counts.csv',
         'FBXL3 Pc - M17_Counts.csv', 'FBXL3 Pc - M18_Counts.csv',
         'FBXL3 Pc - M19_Counts.csv', 'FBXL3 Pc - F20_Counts.csv',
         'FBXL3 Pc - F21_Counts.csv', 'FBXL3 Pc - F22_Counts.csv',
         'FBXL3 Pc - F23_Counts.csv', 'FBXL3 Pc - M24_Counts.csv',
         'FBXL3 Pc - M25_Counts.csv', 'FBXL3 Pc - F26_Counts.csv', 'FBXL3 Pc - M39_Counts.csv',  # 48 rep 1,2,3
         'FBXL3 Pc - F27_Counts.csv', 'FBXL3 Pc - F28_Counts.csv',
         # 'FBXL3 Pc - M40_Counts.csv', # 52 rep 1,2
         #'FBXL3 Pc - F29_Counts.csv',
         'FBXL3 Pc - F30_Counts.csv', 'FBXL3 Pc - M41_Counts.csv',  # 56 rep 2,3
         'FBXL3 Pc - F31_Counts.csv', 'FBXL3 Pc - F32_Counts.csv',
         # 'FBXL3 Pc - M42_Counts.csv', # 60 rep 1,2
         'FBXL3 Pc - M33_Counts.csv', 'FBXL3 Pc - M34_Counts.csv',
         'FBXL3 Pc - M35_Counts.csv', 'FBXL3 Pc - M36_Counts.csv',
         'FBXL3 Pc - M37_Counts.csv', 'FBXL3 Pc - M38_Counts.csv'
         ]

collected_fb_phases2 = []
collected_fb_cc2 = []
collected_fb_periods2 = []
for idx in range(len(FB_rep_times)):
    path = path_root + files[idx]
    FB = load_invivo_data(path, binning=15, sample_days=[4], start_day=[5, 3],
                          infection_day=[5, 4], sample_zt=FB_rep_times[idx])
    FB = cwt_onset(FB, FB_rep_cc[idx], N=5)

    if FB['period'] < 24: print(files[idx])
    elif np.isnan(FB['period']): print(files[idx])
    else:
        collected_fb_phases2.append(FB['sample_phases'])
        collected_fb_cc2.append(FB['cc'])
        collected_fb_periods2.append(FB['period'])

# Collect FB #2
collected_fb_phases2 = np.hstack(collected_fb_phases2)
collected_fb_cc2 = np.hstack(collected_fb_cc2)


#
#   Collect the two days of cell cycle data from WT mice
#   FBXL3 cell cycle dynamics-2.xlsx, sheet "WT 24h resolution ratios"
#   note that this also has the FBXL3 RNAseq experiment
collected_wt_phases2 = []
collected_wt_cc2 = []
collected_wt_periods2 = []

round1_times = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
round1_cc = np.array([0.598, 0.446, 0.453, 0.077, 0.129, 0.644,
                      0.759, 0.692, 0.270, 0.237, 0.197, 0.592])
round1_firstdays = [14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15]
path_root = 'data/figures/wheel_ratios/WT24h_resolution_ratios/ROUND1/'
files = ['WT1_F_old_Counts.csv',
         'WT2_F_old_Counts.csv',
         'WT3_F_old_Counts.csv',
         'WT4_M_old_Counts.csv',
         'WT5_M_old_Counts.csv',
         'WT6_M_old_Counts.csv',
         'WT7_F_young_Counts.csv',
         'WT8_F_young_Counts.csv',
         'WT9_F_young_Counts.csv',
         'WT10_M_young_Counts.csv',
         'WT11_M_young_Counts.csv',
         'WT12_M_young_Counts.csv']
for idx in range(len(round1_times)):
    path = path_root + files[idx]
    fd = round1_firstdays[idx]
    WT = load_invivo_data(path, binning=15, sample_days=[5], start_day=[6, fd],
                          infection_day=[6, 15], sample_zt=round1_times[idx])
    WT = cwt_onset(WT, round1_cc[idx], N=5)

    if WT['period'] > 25:
        print(files[idx])
    elif np.isnan(WT['period']): print(files[idx])
    else:
        collected_wt_phases2.append(WT['sample_phases'])
        collected_wt_cc2.append(WT['cc'])
        collected_wt_periods2.append(WT['period'])

round2_times = np.array([0, 4, 8, 12, 16, 20, 24, 28, 44])
round2_cc = np.array(
    [0.722, 0.580, 0.187, 0.158, 0.133, 0.596, 0.667, 0.673, 0.584])
path_root = 'data/figures/wheel_ratios/WT24h_resolution_ratios/ROUND2/'
files = ['WT1_F_Counts.csv',
         'WT2_F_Counts.csv',
         'WT3_F_Counts.csv',
         'WT4_F_Counts.csv',
         'WT5_F_Counts.csv',
         'WT6_M_Counts.csv',
         'WT7_M_Counts.csv',
         'WT8_M_Counts.csv',
         'WT9_M_Counts.csv']
collected_round2_phases = []
for idx in range(len(round2_times)):
    path = path_root + files[idx]
    WT = load_invivo_data(path, binning=15, sample_days=[5], start_day=[8, 2],
                          infection_day=[8, 3], sample_zt=round2_times[idx])
    WT = cwt_onset(WT, round2_cc[idx], N=5)
    if WT['period'] > 25:
        print(files[idx])
    elif np.isnan(WT['period']): print(files[idx])
    else:
        collected_wt_phases2.append(WT['sample_phases'])
        collected_wt_cc2.append(WT['cc'])
        collected_wt_periods2.append(WT['period'])


WT_frep_times = np.array([0, 0,
                          4, 4,
                          8, 8,
                          12, 12,
                          #
                          20, 20,
                          24, 24,
                          28, 28,
                          32, 32,
                          36, 36,
                          40, 40,
                          44, 44,
                          48, 48,
                          52,
                          56, 56,
                          60, 60,
                          68, 68,
                          72])
WT_frep_cc = np.array([0.621, 0.565,
                       0.584, 0.525,
                       0.392, 0.410,
                       0.315, 0.303,
                       #
                       0.297, 0.308,
                       0.423, 0.405,
                       0.531, 0.587,
                       0.438, 0.400,
                       0.310, 0.270,
                       0.331, 0.181,
                       0.400, 0.323,
                       0.535, 0.511,
                       0.630,
                       0.441, 0.411,
                       0.187, 0.212,
                       #
                       0.269, 0.330,
                       0.419])
path_root = 'data/figures/wheel_ratios/WT24h_resolution_ratios/WTinFBXL3/'
# ignore where missing
files = ['WT Pc- F1_Counts.csv',
         'WT Pc- F2_Counts.csv',
         'WT Pc- F3_Counts.csv',
         'WT Pc- M4_Counts.csv',
         'WT Pc- M5_Counts.csv',
         'WT Pc- M6_Counts.csv',
         'WT Pc- M7_Counts.csv',
         'WT Pc- M8_Counts.csv',
         #'WT Pc- F9_Counts.csv',
         #'WT Pc- F10_Counts.csv',
         'WT Pc- M11_Counts.csv',
         'WT Pc- M12_Counts.csv',
         'WT Pc- F13_Counts.csv',
         'WT Pc- F14_Counts.csv',
         'WT Pc- F15_Counts.csv',
         'WT Pc- F16_Counts.csv',
         'WT Pc- M17_Counts.csv',
         'WT Pc- M18_Counts.csv',
         'WT Pc- M19_Counts.csv',
         'WT Pc- M20_Counts.csv',
         'WT Pc- M21_Counts.csv',
         'WT Pc- M22_Counts.csv',
         'WT Pc- M23_Counts.csv',
         'WT Pc- M24_Counts.csv',
         'WT Pc- M25_Counts.csv',
         'WT Pc- M26_Counts.csv',
         'WT Pc- M27_Counts.csv',
         #'WT Pc- M28_Counts.csv',
         'WT Pc- M29_Counts.csv',
         'WT Pc- M30_Counts.csv',
         'WT Pc- M31_Counts.csv',
         'WT Pc- M32_Counts.csv',
         #'WT Pc- M33_Counts.csv',
         #'WT Pc- M34_Counts.csv',
         'WT Pc- M35_Counts.csv',
         'WT Pc- M36_Counts.csv',
         'WT Pc- M37_Counts.csv']
for idx in range(len(WT_frep_times)):
    path = path_root + files[idx]
    WT = load_invivo_data(path, binning=15, sample_days=[4], start_day=[5, 3],
                          infection_day=[5, 4], sample_zt=WT_frep_times[idx])
    WT = cwt_onset(WT, WT_frep_cc[idx], N=5)

    if WT['period'] > 25: print(files[idx])
    elif np.isnan(WT['period']): print(files[idx])
    else:
        collected_wt_phases2.append(WT['sample_phases'])
        collected_wt_cc2.append(WT['cc'])
        collected_wt_periods2.append(WT['period'])

# Collect WT #2
collected_wt_phases2 = np.hstack(collected_wt_phases2)
collected_wt_cc2 = np.hstack(collected_wt_cc2)


#
#   PART 3: Collect the multiple days and wheel data
# FBXL3 cell cycle dynamics-2.xlsx, sheet "Sheet2"
collected_fb_cc3 = []
collected_fb_phases3 = []
collected_fb_periods3 = []

path_root = 'data/figures/wheel_ratios/Ratio_hormones_exp/'

# for those at ZT2
zt2_times = [2, 2, 2, 2]

# FB ZT2
FBzt2 = [0.472, 0.314, 0.253, 0.522]
wheelday = [2, 2, 2, 2, ]
infectday = [3, 3, 3, 3]
files = ['Pc_FBXL3-1_Counts.csv',
         'Pc_FBXL3-2_Counts.csv',
         'Pc_FBXL3-3_Counts.csv',
         'Pc_FBXL3-4_Counts.csv']
for idx in range(len(zt2_times)):
    path = path_root + files[idx]
    FB2h = load_invivo_data(path, binning=15, sample_days=[6],
                            start_day=[8, wheelday[idx]],
                            infection_day=[8, infectday[idx]],
                            sample_zt=zt2_times[idx])
    FB2h = cwt_onset(FB2h, FBzt2[idx], N=-1)

    if FB2h['period'] < 24: print(files[idx])
    elif np.isnan(FB2h['period']): print(files[idx])
    else:
        collected_fb_phases3.append(FB2h['sample_phases'])
        collected_fb_cc3.append(FB2h['cc'])
        collected_fb_periods3.append(FB2h['period'])


# for those at ZT14
zt14_times = [14, 14, 14, 14]

# FB zt14
FBzt14 = [0.248, 0.214, 0.373, 0.294]
wheelday = [2, 2, 2, 2]
infectday = [3, 3, 3, 3]
files = ['Pc_FBXL3-5_Counts.csv',
         'Pc_FBXL3-6_Counts.csv',
         'Pc_FBXL3-13_Counts.csv',
         'Pc_FBXL3-14_Counts.csv']
for idx in range(len(zt14_times)):
    path = path_root + files[idx]
    FB14 = load_invivo_data(path, binning=15, sample_days=[6],
                            start_day=[8, wheelday[idx]],
                            infection_day=[8, infectday[idx]],
                            sample_zt=zt14_times[idx])
    FB14 = cwt_onset(FB14, FBzt14[idx], N=-1)

    if FB14['period'] < 24: print(files[idx])
    elif np.isnan(FB14['period']): print(files[idx])
    else:
        collected_fb_phases3.append(FB14['sample_phases'])
        collected_fb_cc3.append(FB14['cc'])
        collected_fb_periods3.append(FB14['period'])

# Collect FB #3
collected_fb_cc3 = np.hstack(collected_fb_cc3)
collected_fb_phases3 = np.hstack(collected_fb_phases3)


# for for WT3
collected_wt_cc3 = []
collected_wt_phases3 = []
collected_wt_periods3 = []

# for those at ZT2
zt2_times = [2, 2, 2, 2]

# WT zt2
WTzt2 = [0.775, 0.714, 0.691, 0.693]
wheelday = [2, 2, 2, 2]
infectday = [3, 3, 3, 3]  # relative to JULY
files = ['Pc_WT1_Counts.csv',
         'Pc_WT2_Counts.csv',
         'Pc_WT3_Counts.csv',
         'Pc_WT4_Counts.csv']
for idx in range(len(zt2_times)):
    path = path_root + files[idx]
    WT2h = load_invivo_data(path, binning=15, sample_days=[6],
                            start_day=[7, wheelday[idx]],
                            infection_day=[7, infectday[idx]],
                            sample_zt=zt2_times[idx])
    WT2h = cwt_onset(WT2h, WTzt2[idx], N=-1)

    if WT2h['period'] > 25:
        print(files[idx])
    elif np.isnan(WT2h['period']): print(files[idx])
    else:
        collected_wt_phases3.append(WT2h['sample_phases'])
        collected_wt_cc3.append(WT2h['cc'])
        collected_wt_periods3.append(WT2h['period'])

# for those at ZT14
zt14_times = [14, 14, 14, 14]

# WT ZT14
WTzt14 = [0.128, 0.139, 0.239, 0.146]
wheelday = [2, 2, 2, 2]
infectday = [3, 3, 3, 3]  # relative to JULY
files = ['Pc_WT5_Counts.csv',
         'Pc_WT6_Counts.csv',
         'Pc_WT7_Counts.csv',
         'Pc_WT8_Counts.csv']
for idx in range(len(zt14_times)):
    path = path_root + files[idx]
    WT14 = load_invivo_data(path, binning=15, sample_days=[6],
                            start_day=[7, wheelday[idx]],
                            infection_day=[7, infectday[idx]],
                            sample_zt=zt14_times[idx])
    WT14 = cwt_onset(WT14, WTzt14[idx], N=-1)

    if WT14['period'] > 25:
        print(files[idx])
    elif np.isnan(WT14['period']): print(files[idx])
    else:
        collected_wt_phases3.append(WT14['sample_phases'])
        collected_wt_cc3.append(WT14['cc'])
        collected_wt_periods3.append(WT14['period'])

# Collect WT #3
collected_wt_cc3 = np.hstack(collected_wt_cc3)
collected_wt_phases3 = np.hstack(collected_wt_phases3)


# remove NANs where there was only one onset
total_fb_phases = np.hstack(
    [collected_fb_phases1, collected_fb_phases2, collected_fb_phases3])
total_fb_ccs = np.hstack(
    [collected_fb_cc1, collected_fb_cc2, collected_fb_cc3])
total_fb_ccs = total_fb_ccs[~np.isnan(total_fb_phases)]
total_fb_phases = total_fb_phases[~np.isnan(total_fb_phases)]

total_wt_phases = np.hstack(
    [collected_wt_phases1, collected_wt_phases2, collected_wt_phases3])
total_wt_ccs = np.hstack(
    [collected_wt_cc1, collected_wt_cc2, collected_wt_cc3])
total_wt_ccs = total_wt_ccs[~np.isnan(total_wt_phases)]
total_wt_phases = total_wt_phases[~np.isnan(total_wt_phases)]

total_wt_periods = np.hstack([collected_wt_periods1, collected_wt_periods2,
                              collected_wt_periods3])
total_fb_periods = np.hstack([collected_fb_periods1, collected_fb_periods2,
                              collected_fb_periods3])

# test for rhythms: use only the 2pi period length.
# how would we even test for other period lengths? scale each mouse period
# individually, somehow?
# the reason we can't test for non-2pi periods is because the x-axis is a
# nonlinear function of period, so we can't just scale them all.
wt_thresh = pro.identify_LS_threshold2(
    total_wt_phases,
    1,
    res=3,
    per_low=2 *
    np.pi -
    0.01,
    per_high=2 *
    np.pi +
    0.01)
fb_thresh = pro.identify_LS_threshold2(
    total_fb_phases,
    1,
    res=3,
    per_low=2 *
    np.pi -
    0.01,
    per_high=2 *
    np.pi +
    0.01)
wt_test_data = stats.zscore(total_wt_ccs)
fb_test_data = stats.zscore(total_fb_ccs)

lsp_wt_results = pro.LS_pgram_RNAseq(
    total_wt_phases,
    wt_test_data,
    threshold=wt_thresh,
    res=3,
    per_low=2 * np.pi - 0.01,
    per_high=2 * np.pi + 0.01)
lsp_fb_results = pro.LS_pgram_RNAseq(
    total_fb_phases,
    fb_test_data,
    threshold=wt_thresh,
    res=3,
    per_low=2 * np.pi - 0.01,
    per_high=2 * np.pi + 0.01)

# sinusoidal fitting for the data
wt_sin = pro.fit_sin(total_wt_phases, total_wt_ccs, 2 * np.pi, per_bound=0.01)
fb_sin = pro.fit_sin(total_fb_phases, total_fb_ccs, 2 * np.pi, per_bound=0.01)

# extract params
ts = np.arange(0, 2 * np.pi, 0.01)
wwt = wt_sin['freq']
Awt = wt_sin['amp']
pwt = wt_sin['phase']
cwt = wt_sin['offset']
wfb = fb_sin['freq']
Afb = fb_sin['amp']
pfb = fb_sin['phase']
cfb = fb_sin['offset']


# results and plot for the paper
plo.PlotOptions(ticks='in')
plt.figure(figsize=(4, 1.7))
gs = gridspec.GridSpec(1, 3, width_ratios=(1, 1, 0.3))

ax = plt.subplot(gs[0, 0])
ax.plot(ts, Awt * np.cos(ts - pwt - np.pi) + cwt, 'k')
ax.plot((total_wt_phases+np.pi)%(2*np.pi), total_wt_ccs, marker='o', ls='', color='#708090',
        label='WT', markeredgewidth=0.0)
plt.legend(loc=2)

bx = plt.subplot(gs[0, 1])
bx.plot(ts, Afb * np.cos(ts - pfb - np.pi) + cfb, 'k')
bx.plot((total_fb_phases+np.pi)%(2*np.pi), total_fb_ccs, marker='o', ls='', color='#FDD262',
        label='FB', markeredgewidth=0.0)
plt.legend(loc=2)

cx = plt.subplot(gs[0, 2])
cx.plot(0.75 + 0.5 * np.random.rand(len(total_wt_periods)), total_wt_periods,
        c='#708090', marker='o', ls='', markeredgewidth=0.0)
cx.plot(1.75 + 0.5 * np.random.rand(len(total_fb_periods)), total_fb_periods,
        c='#FDD262', marker='o', ls='', markeredgewidth=0.0)
cx.set_xticks([1, 2])
cx.set_xticklabels(['WT', 'FB'])
cx.set_ylabel('Mouse Activity Period')
cx.set_xlim([0.5, 2.5])

ax.set_ylim([0, 0.85])
bx.set_ylim([0, 0.85])
bx.set_yticklabels([])
plo.format_2pi_axis(ax)
plo.format_2pi_axis(bx)
ax.set_ylabel('Ratio')
plt.tight_layout(**plo.layout_pad)
plt.savefig('results/figure_1/wheel_ratios/ratios_period_paper.pdf')







# results and plot for personal knowledge
plo.PlotOptions(ticks='in')
plt.figure(figsize=(7.0 / 3 * 2, 2.5))
gs = gridspec.GridSpec(1, 2)

ax = plt.subplot(gs[0, 0])
ax.plot(ts, Awt * np.cos(ts - pwt) + cwt, 'k')
ax.plot(collected_wt_phases1, collected_wt_cc1, marker='o', ls='', color='f',
        label='WT Multi-day')
ax.plot(collected_wt_phases2, collected_wt_cc2, marker='o', ls='', color='h',
        label='WT 24h Resolution Ratios')
ax.plot(collected_wt_phases3, collected_wt_cc3, marker='o', ls='', color='i',
        label='WT Ratio Hormones EXP')
plt.legend()

bx = plt.subplot(gs[0, 1])
bx.plot(ts, Afb * np.cos(ts - pfb) + cfb, 'k')
bx.plot(collected_fb_phases1, collected_fb_cc1, marker='o', ls='', color='j',
        label='FB Multi-day')
bx.plot(collected_fb_phases2, collected_fb_cc2, marker='o', ls='', color='k',
        label='FB 24h Resolution Ratios')
bx.plot(collected_fb_phases3, collected_fb_cc3, marker='o', ls='', color='l',
        label='FB Ratio Hormones EXP')
plt.legend()

ax.set_ylim([0, 1.1])
bx.set_ylim([0, 1.1])
plo.format_2pi_axis(ax)
plo.format_2pi_axis(bx)
ax.set_ylabel('Ratio')
bx.set_xlabel('Activity Phase (h)')
plt.tight_layout(**plo.layout_pad)

plt.savefig('results/figure_1/wheel_ratios/ratios_period.pdf')