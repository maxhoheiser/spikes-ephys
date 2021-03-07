get_ipython().run_line_magic("matplotlib", " notebook")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
plt.rcParams['axes.facecolor'] = '#f0f4f7'

import matplotlib.gridspec as gridspec
from matplotlib.legend import Legend
import math

import csv
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.stats import norm


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


from numba import njit

from sync_class import SyncPhenosys
from eda_class import SpikesEDA
from behavior_class import BehaviorAnalysis
from sda_class import SpikesSDA
from report_class import SpikesReport


from synceda_phenosys import *

get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")

#pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 100)


#plt.style.use('ggplot')
#plt.style.use('gadfly.mplstyle')

get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
#plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/master/nma.mplstyle")


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% get_ipython().getoutput("important; }</style>"))")

import warnings
warnings.filterwarnings('ignore')




# ================================================================================================
# specify file path & sessions

windows_folder = r"C:/Users/User/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"
linux_folder = "/home/max/ExpanDrive/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"
mac_folder = "/Users/max/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"

# list -> [ session_name, [ttl_missing_rows], load_spikes, [spikes_trials_skip] ]
se_li = [('JG14_190621', [1900,1931,1996,2058,2127],True,[(0,6),(215,'end')]),
         ('JG14_190619', [111, 2781],False,[(0,1),(259,'end')]),
         ('JG14_190626', [1428, 1824, 1838, 2861, 2910, 3089, 3245, 3430, 3443],False,[(0,1),(276,'end')]),
         ('JG15_190722', [2094, 2574, 2637, 2808, 2831, 3499],False,[(271,'end')]),
         ('JG15_190725', [366,711,1487,1578,1659,2221,2666,2720,2769,2847,3371,3476],False,[(184,'end')]),
         ('JG18a_190814', [405,2621,2693,2770,2959,3015,3029,3038,3048],False,[(307,'end')]),
         ('JG18b_190828', [1744, 2363, 2648, 2701, 2731, 2778,2953,2967],True,[(0,0),(204,'end')]),
         ]



def load_session(session, missing_rows_ttl=[], lo_spikes=False, deselect_trials=[]):
    # load calss and set folder depending on platform
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder + '/' + session
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder + r"/" + session
    elif platform.system() == 'Darwin':
        folder = mac_folder + r"/" + session
        
    sync_obj = SyncPhenosys(session, folder, 7, 1, missing_rows_ttl) 
    behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
    if lo_spikes:
        print(f"{session} -> sda")
        eda_obj = SpikesEDA(behavior_obj)
        sda_obj = SpikesSDA(eda_obj)
        report_obj = SpikesReport(sda_obj)
        session_obj = type('obj', (object,), 
                       {'sync':sync_obj,
                        'behavior':behavior_obj,
                        'eda':eda_obj,
                        'sda': sda_obj,
                        'report': report_obj,
                       })
    else:
        print(f"{session} -> behavior")
        session_obj = type('obj', (object,), 
               {'sync':sync_obj,
                'behavior':behavior_obj,
               })


    return session_obj


# ==========================================================================================
# save & load sessions from pickl functions

# save dict to pickl
def save_all_sessions_dict_pickl(all_sessions_dict):
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder
    elif platform.system() == 'Darwin':
        folder = mac_folder 
    
    for key,value in all_sessions_dict.items():
        file = folder + f"/{key}.pkl"
        with open(file, 'wb') as dump:
            pickle.dump(value, dump, pickle.HIGHEST_PROTOCOL)
            

# load dict from pickl
def load_all_sessions_dict_pickl():
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder
    elif platform.system() == 'Darwin':
        folder = mac_folder 
        
    file = folder + '/all_sessions_dict.pkl'
    with open(file, 'rb') as dump:
        all_sessions_dict = pickle.load(dump)
    return all_sessions_dict



# =======================================================================================
# save figures
def save_fig(name, fig):
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder+"/figures/all_figures"
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder+"\figures\all_figures"
    elif platform.system() == 'Darwin':
        folder = mac_folder+"/figures/all_figures"
    try:
        fig.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')
    except:
        if platform.system() == 'Linux':
            # Linux
            folder = linux_folder+"/figures"
        elif platform.system() == 'Windows':
            # windows
            folder = windows_folder+"\figures"
        elif platform.system() == 'Darwin':
            folder = mac_folder+"/figures"
        fig.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')


def savefig(name,fig):
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder + '/figures' 
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder + r"/figures" 
    elif platform.system() == 'Darwin':
        folder = mac_folder + r"/figures" 

    Path(folder).mkdir(parents=True, exist_ok=True)

    fig.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')








# load alls session

all_sessions_dict = dict()
# load all sessions into dict
for session_name,missing_rows_ttl,load_spikes,spikes_trials_skip in se_li:
    all_sessions_dict[session_name]=load_session(session_name, missing_rows_ttl, load_spikes, spikes_trials_skip)

all_sessions_dict['JG15_190722'].sync.combined_df.loc[(131,32,428,6),"Delta (TTL-CSV)"]=30


all_sessions_trial_info_df = list()
for key,se in all_sessions_dict.items():
    length_df = get_trial_length_dif(se)
    all_sessions_trial_info_df.append(length_df)


for key,se in all_sessions_dict.items():
    fig,ax = plt_dif(se)
    se.sync.save_fig(f"sync_ttl_dif",fig)



fig,ax = plt.subplots()
max_x = 0

for key,se in all_sessions_dict.items():
    se.sync.combined_df['Delta (TTL-CSV)'].plot(label=se.sync.session, alpha=0.8,ax=ax)
    se_max_x = se.sync.combined_df.shape[0]
    if se_max_x > max_x:
        max_x = se_max_x

ax.set_xlabel("Event")
ax.set_ylabel("Time delta [ms]")
ax.legend()

start = 0
stop = se_max_x
range_x = np.arange(0, stop+1500, 500).astype(int)
ax.set_xticks(range_x)
labels_x = range_x.astype(str).tolist()
ax.set_xticklabels(labels_x)

savefig("ttl_delta", fig)



fig,ax = plt.subplots()
max_x = 0
fit_curves_pol = list()

for key,se in all_sessions_dict.items():
    x,y,x_line,y_line = fit_curve(se,'pol')
    label=se.sync.session
    plt_fit_curve(ax,x,y,x_line,y_line,label)
    fit_curves_pol.append([label,x,y,x_line,y_line])
    
ax.set_xlabel("Event")
ax.set_ylabel("Time delta [ms]")
ax.legend(loc=2)

start = 0
stop = se_max_x
range_x = np.arange(0, stop+1500, 500).astype(int)
ax.set_xticks(range_x)
labels_x = range_x.astype(str).tolist()
ax.set_xticklabels(labels_x)

savefig("ttl_delta_fitted_pol", fig)


for cuve in fit_curves_pol:
    print(f"Session: {cuve[0]}, drift: {cuve[4][-1]-cuve[4][0]}")


fig,ax = plt.subplots()
max_x = 0
fit_curves_lin = list()

for key,se in all_sessions_dict.items():
    x,y,x_line,y_line = fit_curve(se,'lin')
    label=se.sync.session
    plt_fit_curve(ax,x,y,x_line,y_line,label)
    fit_curves_lin.append([label,x,y,x_line,y_line])
    
ax.set_xlabel("Event")
ax.set_ylabel("Time delta [ms]")
ax.legend(loc=2)

start = 0
stop = se_max_x
range_x = np.arange(0, stop+1500, 500).astype(int)
ax.set_xticks(range_x)
labels_x = range_x.astype(str).tolist()
ax.set_xticklabels(labels_x)

savefig("ttl_delta_fitted_lin", fig)


labels = []

width = 8
fig = plt.figure(constrained_layout=True,figsize=(width, width * (3 / 5)))
gs = gridspec.GridSpec(ncols=2,
                        nrows=2,
                        width_ratios=[4, 1],
                        #height_ratios=[2, 1],
                        hspace=0.5,  # ,wspace=0.2
                        )
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[:,1])

for key,se in all_sessions_dict.items():
    length_df = get_trial_length_dif(se)
    ax1.plot(length_df["trial length dif [TTL - CSV]"],label=key)
    lines+= ax2.plot(length_df["trial length cor"],label=key)
    labels.append(key)
    

for ax in [ax1,ax2]:
    ax.set_xlabel("Trial")
    #ax.set_ylabel("trial length dif (TTL - CSV) [ms]")
    #ax.legend()

# create legend in axis 3
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.xaxis.get_label(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax3.yaxis.get_label(), visible=False)
# hide frame
ax3.axis('off')

# create legend
pos1 = ax3.get_position() # get the original position 
pos2 = [pos1.x0, pos1.y0, pos1.width, pos1.height] 
ax3.set_position(pos2)
#labels = lines.get_label()
#ax3.legend(label, loc=0)
leg = Legend(ax3, lines, labels,loc='center right')
ax3.add_artist(leg);

# set y label
fig.text(0.04, 0.5, "trial length delta [ms]", va='center', rotation='vertical')

savefig("trial_length_norm", fig)


labels = []
lines=[]

width = 8
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
    lines+=ax1.plot(length_df["trial length dif [TTL - CSV]"],label=key)
    #lines+= ax1.plot(length_df["trial length cor"],label=key)
    labels.append(key)
    
for ax in [ax1,ax2]:
    ax.set_xlabel("Trial")
    ax.set_ylabel("trial length delta [ms]")

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


savefig("trial_length_all", fig)


labels = []
lines=[]


width = 8
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
    lines+= ax1.plot(length_df["trial length cor"],label=key)
    labels.append(key)
    
for ax in [ax1,ax2]:
    ax.set_xlabel("Trial")
    ax.set_ylabel("trial length delta [ms]")

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
ax2.add_artist(leg);


savefig("trial_length_nooutlier", fig)


fig,ax = plt_fit_normdist(length_df["trial length cor"].dropna())
savefig("trial_length_norm", fig)


all_list = list()


fig,ax = plt.subplots()

for length_df in all_sessions_trial_info_df:
    ax = plt_fit_normdist(length_df["trial length cor"].dropna(),ax,norm_fit=False)
    all_list.extend(length_df["trial length cor"].dropna().values)

legend = ax.get_legend()
legend.remove()

ax.set_xlim([-50, 50])

# plot combined fit
mu, std = norm.fit(all_list)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, "k", linestyle='--', linewidth=3,label=f"combined\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")

# center
mu=3.12
std=3.87
#x = np.linspace(-25, 29, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p*1.1, "blue", linestyle='--', linewidth=3,label=f"center\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")

# right
mu=22.04
std=4
p = norm.pdf(x, mu, std)
#x = np.linspace(12, 33, 100)
ax.plot(x, p/6.5, "red", linestyle='--', linewidth=3,label=f"right\nmu:{round_up(mu,2)}\nstd:{round_up(std,2)}")

ax.legend(prop={'size': 14})

savefig("trial_length_norm_all", fig)


for length_df in all_sessions_trial_info_df:
    test_sapiro(length_df["trial length cor"].dropna())





se.sync.combined_df


info_df = se.sync.combined_df.copy()
info_df = info_df.droplevel([0,2,3])
mini = length_df.loc[np.logical_and(length_df["trial length cor"]>14,length_df["trial length cor"]<33)]
not_mini = length_df.loc[np.invert(np.logical_and(length_df["trial length cor"]>14,length_df["trial length cor"]<33))]

qgrid.show_grid(info_df.loc[mini.index])


qgrid.show_grid(mini)


fig,ax = plt.subplots()

ax.hist(mini['trial length cor'], bins=25, density=True, alpha=0.5, label='mini')
ax.hist(not_mini['trial length cor'], bins=25, density=True, alpha=0.5, label='mini')

ax.legend()


event_dif_df = pd.DataFrame(columns=['session','tot. events', 'dif. events', "% dif",
                                    'start', 'cue', 'sound', 'openl.', 
                                     'gamble rw', 'gamble norw', 'safe rw','safe norw',
                                    'no resp.', 'iti', 'end',
                                    ])

for key,se in all_sessions_dict.items():
    dif_ev = np.sum(se.sync.combined_df['TTL Event']get_ipython().getoutput("=se.sync.combined_df['CSV Event'])")
    all_ev = se.sync.combined_df.shape[0]
    per_ev = dif_ev/(all_ev/100)
    
    
    # get specific dif for each event type
    events_dif = list()
    for event_type in ['start', 'cue', 'sound', 'openloop',
                       'right_rw', 'right_norw', 'left_rw', 'left_norw',
                       'no response in time', 'iti', 'end'
                      ]:
        ev_df = se.sync.combined_df.loc[se.sync.combined_df['CSV Event']==event_type]
        per_ev_type = round_up(np.sum(ev_df['TTL Event']get_ipython().getoutput("=ev_df['CSV Event'])/(ev_df.shape[0]/100),2)")
        events_dif.append(per_ev_type)
        
    if se.sync.gamble_side == 'right':
        events_dif_cor = events_dif
    else:
        events_dif_cor = events_dif[:4] + events_dif[6:8] + events_dif[4:6] + events_dif[8:]

    
    event_dif_df.loc[event_dif_df.shape[0]+1,:] = [key,all_ev,dif_ev,round_up(per_ev,2)]+events_dif_cor

event_dif_df


print(event_dif_df.to_latex())


fig, ax = fingerprint_color_map(event_dif_df)

savefig("event_type_dif", fig)



