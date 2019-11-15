# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 2019

@author: john abel

This file finds first the genes that are fw rhythmic, then looks at how their
patterns are over time. First, we need to take and run the analysis to find out which
genes are in fact rhythmic over the timecourse. For the comparison though, we'll want to
set the x-axis to not be time, but mouse activity phase (since we have established that
oscillatory period indeed tracks with mouse period).
"""

# python 3.7 now 

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import gridspec

from local_imports import PlotOptions as plo
from local_imports import Processing as pro
from local_imports import Utilities as util

# run the code to ID the rhythmic genes under each condition
print("Importing expression data to be processed, and activity pattern functions, please be patient.")

def rhythmic_indexes(gset, r1, r2):
    """ finds genes common to pandas df of r1 and r2, and returns their
    index locations in gset"""
    r1 = r1.index.tolist()
    r2 = r2.index.tolist()
    s = gset.index.tolist()

    rhythmic_inds = []
    for gi, gene in enumerate(s):
        if all([gene in r1, gene in r2]):
            rhythmic_inds.append(gi)
    return np.array(rhythmic_inds)

#=============================================================================
# Loading the expression data, or at least setting up the loading
#=============================================================================

# all the data comes from here
path = 'data/figures/expression_phases/'

# rhythmic under DD (fb matched), LD, FB, DD (yy matched), YY
r_DD72 = pd.read_csv(path + 
            'rhythmic/Results 20-28hWTDD_Cycling wFoldChange.txt',
            delimiter='\t', index_col=0)
r_LD72 = pd.read_csv(path + 
            'rhythmic/Results 20-28hWTLD_Cycling wFoldChange.txt',
            delimiter='\t', index_col=0)
r_FB72 = pd.read_csv(path + 
            'rhythmic/Results 20-28hFB_Cycling wFoldChange.txt',
            delimiter='\t', index_col=0)
r_DD48 = pd.read_csv(path + 
            'rhythmic/Results 20-28hWT_Cycling wFoldChange.txt',
            delimiter='\t', index_col=0)
r_YY48 = pd.read_csv(path + 
            'rhythmic/Results 20-28hYY_Cycling wFoldChange.txt',
            delimiter='\t', index_col=0)

# comparisons:
#   LD-DD
#   FB-DD
#   YY-DD (48)

# load expression data - DD, YY, WTFB (combined), LD
dd72_datasheet = pd.read_csv(path+'PcAS_WTDD input.txt',
                           delimiter='\t', index_col=0)
ld72_datasheet = pd.read_csv(path+'PcAS_WTLD input.txt',
                           delimiter='\t', index_col=0)
fb72_datasheet = pd.read_csv(path+'PcAS_FB input.txt',
                           delimiter='\t', index_col=0)
dd48_datasheet = pd.read_csv(path+'PcAS_WT input.txt',
                           delimiter='\t', index_col=0)
yy48_datasheet = pd.read_csv(path+'PcAS_YY input.txt',
                           delimiter='\t', index_col=0)

# ensure genes are all the same between comparison datasets
assert all(fb72_datasheet.index==dd72_datasheet.index), \
            "72h genes do not match"
assert all(ld72_datasheet.index==dd72_datasheet.index), \
            "72h genes do not match"
assert all(yy48_datasheet.index==dd48_datasheet.index), \
            "48h genes do not match"
assert all(dd48_datasheet.index==dd72_datasheet.index), \
            "48-72 genes do not match"

#=============================================================================
# Processing the expression data
#=============================================================================

# go through all genes listed as rhythmic in the four lists and keep only
# those for further analysis that exist in whichever pairs

# ld-dd
lddd72_inds = rhythmic_indexes(dd72_datasheet, r_DD72, r_LD72)
# fb-dd
fbdd72_inds = rhythmic_indexes(dd72_datasheet, r_DD72, r_FB72)
# yy-dd
yydd48_inds = rhythmic_indexes(dd48_datasheet, r_DD48, r_YY48)
# compare dds. dd48-dd72
dddd_inds = rhythmic_indexes(dd48_datasheet, r_DD48, r_DD72)

# and separate the data to get the values.
dd72 = dd72_datasheet.values[:,:-1] # last col is duplicated
ld72 = ld72_datasheet.values
fb72 = fb72_datasheet.values
dd48 = dd48_datasheet.values
yy48 = yy48_datasheet.values

# timpoints of each sample, from 
dd72_ts = np.array([0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28,
                    32, 32, 36, 36, 40, 40, 44, 44, 48, 48, 52, 52, 56, 56, 
                    60, 60, 64, 64, 68, 68, 72])
ld72_ts = np.array([0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28,
                    32, 32, 36, 36, 40, 40, 44, 44, 48, 48, 52, 52, 56, 56, 
                    60, 60, 64, 64, 68, 68, 72, 72])
fb72_ts = np.array([0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28,
                    32, 32, 36, 36, 40, 40, 44, 44, 48, 48, 52, 52, 56, 56, 
                    60, 60, 64, 64, 68, 68, 72, 72])

dd48_ts = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 0, 4, 8, 12, 
                    16, 20, 24, 28, 44])
yy48_ts = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 0, 4, 8, 12,
                    16, 20, 24, 28, 44])

#=============================================================================
# Loading the activity data
#=============================================================================

# get the wheel data for DD72, FB72, DD48 experiments

# DD72 activity
path_dd72 = path+'wheel_activity/WT24h_resolution_ratios/WTinFBXL3/'
wheel_dd72 = ['WT Pc- F1_Counts.csv',  # 0
              'WT Pc- F2_Counts.csv',  # 0  
              'WT Pc- F3_Counts.csv',  # 4
              'WT Pc- M4_Counts.csv',  # 4
              'WT Pc- M5_Counts.csv',  # 8
              'WT Pc- M6_Counts.csv',  # 8
              'WT Pc- M7_Counts.csv',  # 12
              'WT Pc- M8_Counts.csv',  # 12
              'WT Pc- F9_Counts.csv',  # 16
              'WT Pc- F10_Counts.csv',  # 16
              'WT Pc- M11_Counts.csv',  # 20
              'WT Pc- M12_Counts.csv',  # 20
              'WT Pc- F13_Counts.csv',  # 24
              'WT Pc- F14_Counts.csv',  # 24
              'WT Pc- F15_Counts.csv',  # 28
              'WT Pc- F16_Counts.csv', # 28
              'WT Pc- M17_Counts.csv',  # 32
              'WT Pc- M18_Counts.csv',  # 32
              'WT Pc- M19_Counts.csv',  # 36
              'WT Pc- M20_Counts.csv',  # 36
              'WT Pc- M21_Counts.csv',  # 40
              'WT Pc- M22_Counts.csv',  # 40
              'WT Pc- M23_Counts.csv',  # 44
              'WT Pc- M24_Counts.csv',  # 44
              'WT Pc- M25_Counts.csv',  # 48
              'WT Pc- M26_Counts.csv',  # 48
              'WT Pc- M27_Counts.csv',  # 52
              'WT Pc- M28_Counts.csv',  # 52
              'WT Pc- M29_Counts.csv',  # 56
              'WT Pc- M30_Counts.csv',  # 56
              'WT Pc- M31_Counts.csv',  # 60
              'WT Pc- M32_Counts.csv',  # 60
              'WT Pc- M33_Counts.csv',  # 64
              'WT Pc- M34_Counts.csv',  # 64
              'WT Pc- M35_Counts.csv',  # 68
              'WT Pc- M36_Counts.csv',  # 68
              'WT Pc- M37_Counts.csv',  # 72              
              ]

#  FB72 Activity
path_fb72 = path+'wheel_activity/FB24h_resolution_ratios/'
wheel_fb72 = ['FBXL3 Pc - F01_Counts.csv', # 0
                 'FBXL3 Pc - F02_Counts.csv',
                 'FBXL3 Pc - F03_Counts.csv', # 4
                 'FBXL3 Pc - M04_Counts.csv',
                 'FBXL3 Pc - M05_Counts.csv', # 8
                 'FBXL3 Pc - M06_Counts.csv',
                 'FBXL3 Pc - M07_Counts.csv', # 12
                 'FBXL3 Pc - M08_Counts.csv',
                 'FBXL3 Pc - M09_Counts.csv', # 16
                 'FBXL3 Pc - M10_Counts.csv',
                 'FBXL3 Pc - M11_Counts.csv', # 20
                 'FBXL3 Pc - M12_Counts.csv',
                 'FBXL3 Pc - F13_Counts.csv', # 24
                 'FBXL3 Pc - F14_Counts.csv',
                 'FBXL3 Pc - F15_Counts.csv', # 28
                 'FBXL3 Pc - M16_Counts.csv',
                 'FBXL3 Pc - M17_Counts.csv', # 32
                 'FBXL3 Pc - M18_Counts.csv',
                 'FBXL3 Pc - M19_Counts.csv', # 36
                 'FBXL3 Pc - F20_Counts.csv',
                 'FBXL3 Pc - F21_Counts.csv', # 40
                 'FBXL3 Pc - F22_Counts.csv',
                 'FBXL3 Pc - F23_Counts.csv', # 44
                 'FBXL3 Pc - M24_Counts.csv',
                 'FBXL3 Pc - M25_Counts.csv', # 48
                 'FBXL3 Pc - F26_Counts.csv',
                 'FBXL3 Pc - F27_Counts.csv', # 52
                 'FBXL3 Pc - F28_Counts.csv',
                 'FBXL3 Pc - M41_Counts.csv', # note from Filipa 56A<-56C
                 'FBXL3 Pc - F30_Counts.csv',
                 'FBXL3 Pc - F31_Counts.csv', # 60
                 'FBXL3 Pc - F32_Counts.csv',
                 'FBXL3 Pc - M33_Counts.csv', # 64
                 'FBXL3 Pc - M34_Counts.csv',
                 'FBXL3 Pc - M35_Counts.csv', # 68
                 'FBXL3 Pc - M36_Counts.csv',
                 'FBXL3 Pc - M37_Counts.csv', # 72
                 'FBXL3 Pc - M38_Counts.csv']

# dd48 for the dd vs. yy data
path_dd48 = path+'wheel_activity/WT24h_resolution_ratios/'
wheel_dd48 = ['ROUND1/WT1_F_old_Counts.csv',  # 0h
              'ROUND1/WT2_F_old_Counts.csv',  # 4h
              'ROUND1/WT3_F_old_Counts.csv',  # 8h
              'ROUND1/WT4_M_old_Counts.csv',  # 12h
              'ROUND1/WT5_M_old_Counts.csv',  # 16h
              'ROUND1/WT6_M_old_Counts.csv',  # 20h
              'ROUND1/WT7_F_young_Counts.csv',  # 24h
              'ROUND1/WT8_F_young_Counts.csv',  # 28h
              'ROUND1/WT9_F_young_Counts.csv',  # 32h
              'ROUND1/WT10_M_young_Counts.csv',  # 36h
              'ROUND1/WT11_M_young_Counts.csv',  # 40h
              'ROUND1/WT12_M_young_Counts.csv',  # 44h
              'ROUND2/WT1_F_Counts.csv',  # 0h
              'ROUND2/WT2_F_Counts.csv',  # 4h
              'ROUND2/WT3_F_Counts.csv',  # 8h
              'ROUND2/WT4_F_Counts.csv',  # 12h
              'ROUND2/WT5_F_Counts.csv',  # 16h
              'ROUND2/WT6_M_Counts.csv',  # 20h
              'ROUND2/WT7_M_Counts.csv',  # 24h
              'ROUND2/WT8_M_Counts.csv',  # 28h
              'ROUND2/WT9_M_Counts.csv'  # 44h
              ]

#=============================================================================
# Processing the wheel data
#=============================================================================

# DD72
dd72_activity_phases = []
wt_mice = 0
for didx, df in enumerate(wheel_dd72):
    file = path_dd72 + df
    WT = pro.load_invivo_data(
        file, binning=15, sample_days=[4], start_day=[5, 3],
                          infection_day=[5, 4], sample_zt=dd72_ts[didx])
    WT = pro.cwt_onset(WT, 0, N=5)
    if all([WT['period'] < 25., WT['period'] >20]):
        dd72_activity_phases.append(WT['sample_phases'])
        wt_mice +=1
    else:
        print(WT['period'])
        dd72_activity_phases.append(np.array([np.nan]))
dd72_activity_phases = np.array(dd72_activity_phases).flatten()


# FB72
fb72_activity_phases = []
fb_mice = 0
for didx, df in enumerate(wheel_fb72):
    file = path_fb72 + df
    FB = pro.load_invivo_data(
        file, binning=15, sample_days=[4], start_day=[5, 3],
                          infection_day=[5, 4], sample_zt=fb72_ts[didx])
    FB = pro.cwt_onset(FB, 0, N=5)
    if all([FB['period'] < 30., FB['period'] >24]):
        fb72_activity_phases.append(FB['sample_phases'])
        fb_mice +=1
    else:
        print(FB['period'])
        fb72_activity_phases.append(np.array([np.nan]))
fb72_activity_phases = np.array(fb72_activity_phases).flatten()


# DD48
dd48_activity_phases = []
fd = 0
for didx, df in enumerate(wheel_dd48):
    file = path_dd48 + df
    # switch between Round 1 and Round 2
    if df[5] == '1':
        round1_firstdays = [14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15]
        WT = pro.load_invivo_data(
            file, binning=15, sample_days=[5], start_day=[
                6, fd], infection_day=[
                6, 15], sample_zt=dd72_ts[didx])
        fd += 1  # increment the first day ticker
    elif df[5] == '2':
        WT = pro.load_invivo_data(
            file, binning=15, sample_days=[5], start_day=[
                8, 2], infection_day=[
                8, 3], sample_zt=dd48_ts[didx])
    WT = pro.cwt_onset(WT, 0, N=5)
    if all([WT['period'] < 25., WT['period'] >20]):
        dd48_activity_phases.append(WT['sample_phases'])
    else:
        dd48_activity_phases.append(np.nan)
        print(WT['period'])
dd48_activity_phases = np.array(dd48_activity_phases).flatten()

#=============================================================================
# Fitting the sinusoids to data
#=============================================================================

# fit sinusoids to every gene, with a period restricted to 2pi
# from this get phases, and see if there's some consistency


# ld72 vs dd72

# get ld phases
ld72_vsdd_phases = []
ld72_vsdd_amps = []
common_lddd = ld72[lddd72_inds, :]
for idx, datai in enumerate(common_lddd):
    sin_fit = pro.fit_sin(ld72_ts/24*2*np.pi, datai,
                          2 * np.pi, per_bound=0.01)
    ld72_vsdd_phases.append(sin_fit['phase'])
    ld72_vsdd_amps.append(sin_fit['amp'])

# get dd phases
dd72_vsld_phases = []
dd72_vsld_amps = []
common_ddld = dd72[lddd72_inds, :]
for idx, datai in enumerate(common_ddld):
    sin_fit = pro.fit_sin(dd72_activity_phases, datai,
                          2 * np.pi, per_bound=0.01)
    dd72_vsld_phases.append(sin_fit['phase'])
    dd72_vsld_amps.append(sin_fit['amp'])

# wrap onto 0,2pi from -2pi,2pi (better for sin fit)
ld72_vsdd_phases = np.array(ld72_vsdd_phases) % (2 * np.pi)
dd72_vsld_phases = np.array(dd72_vsld_phases) % (2 * np.pi)


# fb72 vs dd72

# get fb phases
fb72_vsdd_phases = []
fb72_vsdd_amps = []
common_fbdd = fb72[fbdd72_inds, :]
for idx, datai in enumerate(common_fbdd):
    sin_fit = pro.fit_sin(fb72_activity_phases, datai,
                          2 * np.pi, per_bound=0.01)
    fb72_vsdd_phases.append(sin_fit['phase'])
    fb72_vsdd_amps.append(sin_fit['amp'])

# get dd phases
dd72_vsfb_phases = []
dd72_vsfb_amps = []
common_ddfb = dd72[fbdd72_inds, :]
for idx, datai in enumerate(common_ddfb):
    sin_fit = pro.fit_sin(dd72_activity_phases, datai,
                          2 * np.pi, per_bound=0.01)
    dd72_vsfb_phases.append(sin_fit['phase'])
    dd72_vsfb_amps.append(sin_fit['amp'])

# wrap onto 0,2pi from -2pi,2pi (better for sin fit)
fb72_vsdd_phases = np.array(fb72_vsdd_phases) % (2 * np.pi)
dd72_vsfb_phases = np.array(dd72_vsfb_phases) % (2 * np.pi)


# yy48 vs dd48

# get yy phases
yy48_vsdd_phases = []
yy48_vsdd_amps = []
common_yydd = yy48[yydd48_inds, :]
yy_period = 24.1 # approx, given transcription data
for idx, datai in enumerate(common_yydd):
    sin_fit = pro.fit_sin(yy48_ts/yy_period*2*np.pi, datai,
                          2 * np.pi, per_bound=0.01)
    yy48_vsdd_phases.append(sin_fit['phase'])
    yy48_vsdd_amps.append(sin_fit['amp'])

# get dd phases
dd48_vsyy_phases = []
dd48_vsyy_amps = []
common_ddyy = dd48[yydd48_inds, :]
for idx, datai in enumerate(common_ddyy):
    sin_fit = pro.fit_sin(dd48_activity_phases, datai,
                          2 * np.pi, per_bound=0.01)
    dd48_vsyy_phases.append(sin_fit['phase'])
    dd48_vsyy_amps.append(sin_fit['amp'])

# wrap onto 0,2pi from -2pi,2pi (better for sin fit)
dd48_vsyy_phases = np.array(dd48_vsyy_phases) % (2 * np.pi)
yy48_vsdd_phases = np.array(yy48_vsdd_phases) % (2 * np.pi)

# dd72 vs dd48

# get dd72 phases
dd72_vsdd_phases = []
dd72_vsdd_amps = []
common_dd72dd = dd72[dddd_inds, :]
for idx, datai in enumerate(common_dd72dd):
    sin_fit = pro.fit_sin(dd72_activity_phases[6:], datai[6:],
                          2 * np.pi, per_bound=0.01)
    dd72_vsdd_phases.append(sin_fit['phase'])
    dd72_vsdd_amps.append(sin_fit['amp'])

# get dd48 phases
dd48_vsdd_phases = []
dd48_vsdd_amps = []
common_dd48dd = dd48[dddd_inds, :]
for idx, datai in enumerate(common_dd48dd):
    sin_fit = pro.fit_sin(dd48_activity_phases, datai,
                          2 * np.pi, per_bound=0.01)
    dd48_vsdd_phases.append(sin_fit['phase'])
    dd48_vsdd_amps.append(sin_fit['amp'])

# wrap onto 0,2pi from -2pi,2pi (better for sin fit)
dd48_vsdd_phases = np.array(dd48_vsdd_phases) % (2 * np.pi)
dd72_vsdd_phases = np.array(dd72_vsdd_phases) % (2 * np.pi)

#=============================================================================
# Plotting the results
#=============================================================================


# It is confirmed that x-axis is the phase of the peak

# 1. the LD vs. DD

# correct such that a phase of 0 is at lights-off
ld72_vsdd_phases_corr = (ld72_vsdd_phases - 2*np.pi*12/24)%(2*np.pi)

# plotting
plo.PlotOptions(ticks='in')
fig = plt.figure(figsize=(7.0, 2.0))
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0, 0])
ax.hist(dd72_vsld_phases, bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#708090', label='WT DD')
ax.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(ax)

bx = plt.subplot(gs[0, 1])
bx.hist(ld72_vsdd_phases_corr, 
        bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#ABDDDE', label='WT LD')
bx.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(bx)

cx = plt.subplot(gs[0, 2])
cx.plot(dd72_vsld_phases, ld72_vsdd_phases_corr, 'ko', alpha=0.1)
cx.plot(np.arange(0, 2 * np.pi, 0.01), np.arange(0, 2 * np.pi, 0.01), c='r')
plo.format_2pi_axis(cx, y=True)

bx.set_xlabel('Phase of Expression Peak')
ax.set_ylabel('Number of Transcripts')
cx.set_ylabel('LD Phase')
cx.set_xlabel('WT DD Phase')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/figure_1/expression_phases/LD_DD.pdf')
plt.close(fig)

# find phase offset
def fitfunc(offset):
    err = dd72_vsld_phases - (ld72_vsdd_phases_corr + offset) % (2 * np.pi)
    return np.sum(err**2)


offsets = np.arange(-np.pi, np.pi, 0.0001)
fits = [fitfunc(oo) for oo in offsets]
fitidx = np.argmin(fits)
offset = offsets[fitidx]
offset_h_LD = offset * 24 / (2 * np.pi)
# should be 0



# 2. the FB vs. DD

# plotting
fig = plt.figure(figsize=(7.0, 2.0))
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0, 0])
ax.hist((dd72_vsfb_phases+np.pi)%(2*np.pi), bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#708090', label='WT DD')
ax.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(ax)

bx = plt.subplot(gs[0, 1])
bx.hist((fb72_vsdd_phases+np.pi)%(2*np.pi), bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#FDD262', label='FB')
bx.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(bx)

cx = plt.subplot(gs[0, 2])
cx.plot((dd72_vsfb_phases+np.pi)%(2*np.pi), (fb72_vsdd_phases+np.pi)%(2*np.pi), 'ko', alpha=0.1)
cx.plot(np.arange(0, 2 * np.pi, 0.01), np.arange(0, 2 * np.pi, 0.01), c='r')
plo.format_2pi_axis(cx, y=True)

bx.set_xlabel('Phase of Expression Peak')
ax.set_ylabel('Number of Transcripts')
cx.set_ylabel('FB Phase')
cx.set_xlabel('WT DD Phase')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/figure_1/expression_phases/FB_DD.pdf')
plt.close(fig)

# find phase offset
def fitfunc(offset):
    err = dd72_vsfb_phases - (fb72_vsdd_phases + offset) % (2 * np.pi)
    return np.sum(err**2)


offsets = np.arange(-np.pi, np.pi, 0.0001)
fits = [fitfunc(oo) for oo in offsets]
fitidx = np.argmin(fits)
offset = offsets[fitidx]
offset_h_FB = offset * 25.5 / (2 * np.pi)
# about 2.62


# 3. YY vs DD

# find the best alignment
def fitfunc(offset):
    err = dd48_vsyy_phases - (yy48_vsdd_phases + offset) % (2 * np.pi)
    return np.sum(err**2)

offsets = np.arange(-np.pi, np.pi, 0.0001)
fits = [fitfunc(oo) for oo in offsets]
fitidx = np.argmin(fits)
yy_offset = offsets[fitidx]

# get it to line up
yy48_vsdd_phases_corr = (yy48_vsdd_phases+yy_offset)%(2*np.pi)

# plot result
fig = plt.figure(figsize=(7.0, 2.0))
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0, 0])
ax.hist(dd48_vsyy_phases, bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#273046', label='WT DD')
ax.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(ax)

bx = plt.subplot(gs[0, 1])
bx.hist(yy48_vsdd_phases_corr, 
        bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#FD6467', label='YY')
bx.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(bx)

cx = plt.subplot(gs[0, 2])
cx.plot(dd48_vsyy_phases, yy48_vsdd_phases_corr, 'ko', alpha=0.1)
cx.plot(np.arange(0, 2 * np.pi, 0.01), np.arange(0, 2 * np.pi, 0.01), c='r')
plo.format_2pi_axis(cx, y=True)

bx.set_xlabel('Phase of Expression Peak')
ax.set_ylabel('Number of Transcripts')
cx.set_ylabel('YY Relative Phase')
cx.set_xlabel('WT DD Phase')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/figure_1/expression_phases/YY_DD.pdf')
plt.close(fig)


# 4. DD48 vs DD72

# find the best alignment
def fitfunc(offset):
    err = dd48_vsdd_phases - (dd72_vsdd_phases + offset) % (2 * np.pi)
    return np.sum(err**2)

offsets = np.arange(-np.pi, np.pi, 0.0001)
fits = [fitfunc(oo) for oo in offsets]
fitidx = np.argmin(fits)
dddd_offset = offsets[fitidx]

# get it to line up
dd72_vsdd_phases_corr = (dd72_vsdd_phases+dddd_offset)%(2*np.pi)

# plot result
fig = plt.figure(figsize=(7.0, 2.0))
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0, 0])
ax.hist(dd48_vsdd_phases, bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#123456', label='DD48')
ax.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(ax)

bx = plt.subplot(gs[0, 1])
bx.hist(dd72_vsdd_phases_corr, 
        bins=np.arange(0, 2 * np.pi + 0.01, 0.04 * np.pi),
        color='#654321', label='DD72')
bx.set_ylim([0, 200])
plt.legend()
plo.format_2pi_axis(bx)

cx = plt.subplot(gs[0, 2])
cx.plot(dd48_vsdd_phases, dd72_vsdd_phases_corr, 'ko', alpha=0.1)
cx.plot(np.arange(0, 2 * np.pi, 0.01), np.arange(0, 2 * np.pi, 0.01), c='r')
plo.format_2pi_axis(cx, y=True)

bx.set_xlabel('Phase of Expression Peak')
ax.set_ylabel('Number of Transcripts')
cx.set_ylabel('DD72 Phase')
cx.set_xlabel('DD48 Phase')

plt.tight_layout(**plo.layout_pad)
#plt.savefig('results/figure_1/expression_phases/DD48_DD72_corr.pdf')
#plt.close(fig)




# additional double plot of DD-DD
plt.figure()
cx = plt.subplot()
cx.plot(dd48_vsdd_phases, dd72_vsdd_phases, 'ko', alpha=0.1)
cx.plot(dd48_vsdd_phases+(2*np.pi), dd72_vsdd_phases, 'ko', alpha=0.1)
cx.plot(dd48_vsdd_phases, dd72_vsdd_phases+(2*np.pi), 'ko', alpha=0.1)
cx.plot(dd48_vsdd_phases+(2*np.pi), dd72_vsdd_phases+(2*np.pi), 'ko', alpha=0.1)
cx.plot(np.arange(0, 4 * np.pi, 0.01), np.arange(0, 4 * np.pi, 0.01), c='r')
plo.format_4pi_axis(cx, y=True)
plt.xlabel('DD48')
plt.ylabel('DD72')
plt.tight_layout(**plo.layout_pad)