"""Set of functions for analyzing and plotting syncronization between the
Phenosys and Intan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
plt.rcParams['axes.facecolor'] = '#f0f4f7'
plt.rc('legend', frameon=True,fancybox=True, framealpha=1)
blue = '#4C72B0'
green = '#55A868'
red = '#C44E52'
purple = '#8172B2'
yellow = '#CCB974'
lightblue = '#64B5CD'

import matplotlib.gridspec as gridspec
from matplotlib.legend import Legend
import math

import csv
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.stats import norm


import itertools
import importlib
import os
import sys
import platform
import qgrid
import datetime
from scipy.interpolate import make_interp_spline, BSpline
import pickle
import random
from pathlib import Path

from matplotlib.colors import LogNorm
import copy
import matplotlib as mpl
import matplotlib.patheffects as path_effects


# Helper Functions =========================================================================================

default = [6.4, 4.8]


# TTL Time Differenze analysis =============================================================================

def plt_dif(se):
    fig,ax = plt.subplots()
    combined = se.sync.combine_dataframes(align=True,rows_missing_ttl=se.sync.rows_missing_ttl)
    combined['Delta (TTL-CSV)'].plot()
    ax.set_xlabel("Event")
    ax.set_ylabel("Time delta [ms]")

    #start = combined['Delta (TTL-CSV)'].min()
    start = 0
    stop = combined['Delta (TTL-CSV)'].max()
    range_y = np.arange(0, stop+200, 200).astype(int)
    ax.set_yticks(range_y)
    label = range_y.astype(str).tolist()
    ax.set_yticklabels(label)
        
    return fig,ax

# define the true objective function
def objective_pol(x, a, b, c):
    return a * x + b * x**2 + c

# define the true objective function
def objective_lin(x, a, b):
    return a * x + b


def fit_curve(se,fit):
    # get data
    delta  =se.sync.combined_df['Delta (TTL-CSV)'].copy()
    delta.reset_index(drop=True,inplace=True)
    delta.dropna(inplace=True)
    
    # choose the input and output variables
    y = delta.values
    x = np.arange(delta.shape[0])
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), 1)
    
    # curve fit
    if fit=='pol':
        popt, _ = curve_fit(objective_pol, x, y)
        # summarize the parameter values
        a, b, c = popt
        print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
        # calculate the output for the range
        y_line = objective_pol(x_line, a, b, c)
        
    if fit=='lin':
        popt, _ = curve_fit(objective_lin, x, y)
        a, b = popt
        print('y = %.5f * x + %.5f' % (a, b))
        # calculate the output for the range
        y_line = objective_lin(x_line, a, b)
        
    return x,y, x_line,y_line
    
def plt_fit_curve(ax,x,y,x_line,y_line,label):
    # plot input vs output
    ax.plot(x, y,color='grey',alpha=0.3)


    # create a line plot for the mapping function
    ax.plot(x_line, y_line,label=label,linewidth=2)#, color='red')
    
# Trial length =====================================================================================================

def plt_trial_length(all_sessions_dict,figsize=default):
    fig,ax = plt.subplots(1,1,figsize=figsize)
    max_x = 0

    for key,se in all_sessions_dict.items():
        se.sync.combined_df['Delta (TTL-CSV)'].plot(label=se.sync.session, alpha=0.8,ax=ax)
        se_max_x = se.sync.combined_df.shape[0]
        if se_max_x > max_x:
            max_x = se_max_x

    ax.set_xlabel("Event")
    ax.set_ylabel("Time delta [ms]")
    ax.legend(frameon=True,fancybox=True, framealpha=1)

    start = 0
    stop = se_max_x
    range_x = np.arange(0, stop+1500, 500).astype(int)
    ax.set_xticks(range_x)
    labels_x = range_x.astype(str).tolist()
    ax.set_xticklabels(labels_x)

    return fig,ax


# Trial length diff ================================================================================================
    
def get_trial_length_dif(se):

    df = se.sync.combined_df.copy()
    df = df.droplevel([0,2,3], axis=0)

    trial_length = np.zeros([int(se.sync.combined_df.index.get_level_values(1).dropna().values.max())+1,2])


    for trial in se.sync.combined_df.index.get_level_values(1).dropna().values.astype(int):
        trial_df = df.loc[trial,:]
        ttl_start = trial_df.loc[trial_df['CSV Event']=='start','TTL Start norm']
        ttl_end = trial_df.loc[trial_df['CSV Event']=='end','TTL Start norm']
        csv_start = trial_df.loc[trial_df['CSV Event']=='start','CSV Start norm']
        csv_end = trial_df.loc[trial_df['CSV Event']=='end','CSV Start norm']

        trial_length[trial]=ttl_end-ttl_start, csv_end-csv_start

    # create data frame
    length_df = pd.DataFrame(trial_length, columns=["ttl length","csv length"])
    # get length diff
    length_df["trial length dif [TTL - CSV]"]=length_df["ttl length"]-length_df["csv length"]
    # correct for outlier
    length_df["trial length cor"] = length_df["trial length dif [TTL - CSV]"]
    length_df.loc[length_df["trial length cor"]>1000]=np.nan
    #calculate z-scores of df
    mean = length_df["trial length cor"].mean()
    std = length_df["trial length cor"].std()

    selector = np.logical_or(
                            length_df["trial length cor"]>mean+3*std,
                            length_df["trial length cor"]<mean-3*std,
                            )
    length_df.loc[(selector),"trial length cor"]=np.nan
    
    
    return length_df


def plt_trial_length_dif(all_sessions_dict,outlier=False,figsize=default):
    labels = []
    lines = []

    width = 8#figsize[0] #8
    fig = plt.figure(constrained_layout=True,figsize=(width, width * (1 / 4)))
    gs = gridspec.GridSpec(ncols=2,
                            nrows=1,
                            width_ratios=[4, 1],
                            hspace=0.5,  # ,wspace=0.2
                            )
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    for key,se in all_sessions_dict.items():
        length_df = get_trial_length_dif(se)
        if outlier:
            lines+=ax1.plot(length_df["trial length dif [TTL - CSV]"],label=key)
        else: 
            lines+= ax1.plot(length_df["trial length cor"],label=key)
        labels.append(key)
        
    for ax in [ax1,ax2]:
        ax.set_xlabel("Trial")
        ax.set_ylabel("Trial length delta [ms]")

    # create legend in axis 3
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.xaxis.get_label(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.yaxis.get_label(), visible=False)
    # hide frame
    ax2.axis('off')

    # create legend
    pos1 = ax2.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0, pos1.width, pos1.height] 
    ax2.set_position(pos2)
    leg = Legend(ax2, lines, labels,loc='center right')
    ax2.add_artist(leg)

    return fig,ax



# Normal Distribution of trial length ============================================================================

def plt_fit_normdist(data,ax=False,norm_fit=True,figsize=[6.4,4.8]):
    """plot normaldistributin fitted to histogram

    Args:
        data (np ar): input data[samples,features]
    """
    mu, std = norm.fit(data)
    
    ret_all=False
    if ax == False:
        ret_all = True
        fig, ax = plt.subplots(1,1,figsize=figsize)
        # plot histogram
        ax.hist(data, bins=25, density=True, alpha=0.6, rasterized=True)
    else:
        ax.hist(data, bins=25, density=True, alpha=0.6, rasterized=True)
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    if norm_fit:
        ax.plot(x, p, "k", linestyle='--', linewidth=1, label=f"combined\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")
        # plot neben gaussian
        # center
        mu=3.12
        std=4
        p = norm.pdf(x, mu, std)
        ax.plot(x, p*0.8, blue, linestyle='-', linewidth=2,label=f"center\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")

        # right
        mu=22.04
        std=4
        p = norm.pdf(x, mu, std)
        ax.plot(x, p*0.1, red, linestyle='-', linewidth=2,label=f"right\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")
    
    
    title = "Normal distribution fitted to data \n(mu:%.2f, std:%.2f)" % (mu, std)

    # namings usw
    ax.set_xlabel("Trial length difference [ms]")
    ax.set_ylabel("Probability")
    ax.legend() #prop={'size': 14}
    #ax.set_title(title)
    
    if ret_all:
        return fig,ax
    else:
        return ax

from scipy.stats import shapiro
def test_sapiro(data):
    # normality test
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
        
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier




# Individual Event Analysis =============================================================================

def get_event_df(all_sessions_dict):
    event_dif_df = pd.DataFrame(columns=['session','tot. events', 'dif. events', "% dif",
                                    'start', 'cue', 'sound', 'openl.', 
                                     'gamble rw', 'gamble norw', 'safe rw','safe norw',
                                    'no resp.', 'iti', 'end',
                                    ])

    for key,se in all_sessions_dict.items():
        dif_ev = np.sum(se.sync.combined_df['TTL Event']!=se.sync.combined_df['CSV Event'])
        all_ev = se.sync.combined_df.shape[0]
        per_ev = dif_ev/(all_ev/100)
        
        
        # get specific dif for each event type
        events_dif = list()
        for event_type in ['start', 'cue', 'sound', 'openloop',
                        'right_rw', 'right_norw', 'left_rw', 'left_norw',
                        'no response in time', 'iti', 'end'
                        ]:
            ev_df = se.sync.combined_df.loc[se.sync.combined_df['CSV Event']==event_type]
            per_ev_type = round_up(np.sum(ev_df['TTL Event']!=ev_df['CSV Event'])/(ev_df.shape[0]/100),2)
            events_dif.append(per_ev_type)
            
        if se.sync.gamble_side == 'right':
            events_dif_cor = events_dif
        else:
            events_dif_cor = events_dif[:4] + events_dif[6:8] + events_dif[4:6] + events_dif[8:]

        
        event_dif_df.loc[event_dif_df.shape[0]+1,:] = [key,all_ev,dif_ev,round_up(per_ev,2)]+events_dif_cor

    return event_dif_df


def fingerprint_color_map(event_dif_df,figsize=default):
    x_labels = event_dif_df.iloc[:,4:].columns.values.tolist()
    y_labels = event_dif_df['session'].values.tolist()

    data = event_dif_df.iloc[:,4:].values.astype(float)

    fig, ax = plt.subplots(1,1,figsize=figsize)
    #im = ax.imshow(data,norm=LogNorm())

    my_cmap = copy.copy(mpl.cm.get_cmap('viridis')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    im = ax.imshow(data, 
               norm=mpl.colors.LogNorm(), 
               interpolation='nearest', 
               cmap=my_cmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.


    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")
            text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                           path_effects.Normal()])

    #ax.set_title("Distribution of Event difference")
    #fig.colorbar(im)
    ax.grid(False)

    fig.tight_layout()
    return fig,ax


# Boxplots ==============================================================================================================
def convert_numeric(columns, df):
    for column in columns:
        df[column] = pd.to_numeric(df[column])
    return df

def get_trial_info(all_sessions_dict):
    session_info_df = pd.DataFrame(columns=['id','blocks','tot. trials','wheel ns trials','no resp trials','selected trials',
                                           'reward','no-reward','gamble', 'safe',
                                           'gamble reward','safe reward','gamble no-reward','safe no-reward',
                                           ],
                                  )
    # get individual session objects
    for key,value in all_sessions_dict.items():
        session_name=key
        gamble_side=value.behavior.gamble_side
        missing_rows_ttl=value.sync.rows_missing_ttl
        combined=value.sync.combined_df
        trials=value.sync.all_trials_df
        
        # get info of combined_df
        prop_bin=list((combined['CSV Probability']).dropna().unique())
        # get info form combined
        all_trials = int(combined.index.get_level_values(0).max())
        good_trials = int(combined.index.get_level_values(1).max())
        wheel_ns_trials = all_trials - good_trials
        no_resp_trials = value.behavior.good_trials_df[value.behavior.good_trials_df['event']=='no response in time'].shape[0]
        selected_trials = value.behavior.selected_trials_df.shape[0]
        
        rw=(value.behavior.selected_trials_df["reward_given"]==True).sum()
        norw=(value.behavior.selected_trials_df["reward_given"]==False).sum()
        
        right = (value.behavior.selected_trials_df["right"]==True).sum()
        right_rw = (value.behavior.selected_trials_df[(value.behavior.selected_trials_df["right"])&(value.behavior.selected_trials_df["reward_given"])]).shape[0]
        right_norw = (value.behavior.selected_trials_df[(value.behavior.selected_trials_df["right"])&np.invert(value.behavior.selected_trials_df["reward_given"])]).shape[0]
        left = (value.behavior.selected_trials_df["left"]==True).sum()
        left_rw = (value.behavior.selected_trials_df[(value.behavior.selected_trials_df["left"])&(value.behavior.selected_trials_df["reward_given"])]).shape[0]
        left_norw = (value.behavior.selected_trials_df[(value.behavior.selected_trials_df["left"])&np.invert(value.behavior.selected_trials_df["reward_given"])]).shape[0]
        
        # set safe and gamble side
        if gamble_side=="right":
            gamble=right
            gamble_rw=right_rw
            gamble_norw=right_norw
            safe=left
            safe_rw=left_rw
            safe_norw=left_norw
        elif gamble_side=="left":
            gamble=left
            gamble_rw=left_rw
            gamble_norw=left_norw
            save=right
            save_rw=right_rw    
            save_norw=right_norw

        session_info_df.loc[session_info_df.shape[0] + 1] = [session_name,
                                                             prop_bin,
                                                             all_trials,
                                                             wheel_ns_trials,
                                                             no_resp_trials,
                                                             selected_trials,
                                                             rw,
                                                             norw,
                                                             gamble,
                                                             safe,
                                                             gamble_rw,
                                                             safe_rw,
                                                             gamble_norw,
                                                             safe_norw
                                                             ]
        
        session_info_df = convert_numeric(['tot. trials','wheel ns trials','no resp trials','selected trials',
                                           'reward','no-reward','gamble', 'safe',
                                           'gamble reward','safe reward', 'gamble no-reward', 'safe no-reward'],
                       session_info_df)
        
    return session_info_df

def boxplot(df,columns,scatter=True,title=None,figsize=default):
    fig,ax = plt.subplots(1,1,figsize=figsize)
    df.boxplot(column=columns,return_type='axes',ax=ax)

    marker = itertools.cycle(("X", 'o', '*', "s"))

    if scatter:
        for row in df.index:
            y=df.loc[row,columns]
            x=np.arange(1,len(columns)+1)
            ax.plot(x, y, linestyle='',marker=next(marker),label=df.loc[row,'id'],markersize=10 )
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.legend() #prop={'size': 12}
    return fig,ax