


get_ipython().run_line_magic("matplotlib", " agg")
get_ipython().run_line_magic("matplotlib", " agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
plt.rcParams['axes.facecolor'] = '#f0f4f7'

import csv
import scipy.stats as st
import importlib
import os
import sys
import platform
import qgrid
import datetime
from scipy.interpolate import make_interp_spline, BSpline
import pickle
import random

from numba import njit

from sync_class import SyncPhenosys
from eda_class import SpikesEDA
from behavior_class import BehaviorAnalysis
from sda_class import SpikesSDA
from report_class import SpikesReport


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

      
    
# =======================================================================================
@njit(fastmath=True)
# binn data
def bin_trial_spike_times_all_cluster(input_ar,nr_bins):
    """binn randm windows from all clusters, all trials all iterations over complete trial

    Args:
        input_ar (np ar): spikes per random event for all clusters, all trials, all iterations
        nr_bins (int): number to bin trial

    Returns:
        np ar: array of binns (i=cluster,j=bin,k=iteration, data=bin count)
    """
    cluster=input_ar.shape[0]
    iterations=input_ar.shape[2]
    # y = cluster index
    # x = bin number 1 to 50
    # z = random iteration 1 to 1000
    data_ar=np.zeros(shape=(cluster,nr_bins,iterations),dtype=int)
    for cl in range(cluster):
        for it in range(iterations):
            data_ar[cl,:,it]=(np.histogram(np.concatenate(input_ar[cl,:,it]).ravel(),bins=nr_bins))[0]
    return data_ar




# =======================================================================================
@njit(fastmath=True)
# binn data
def bin_trial_spike_times_all_cluster(input_ar,nr_bins):
    """binn randm windows from all clusters, all trials all iterations over complete trial

    Args:
        input_ar (np ar): spikes per random event for all clusters, all trials, all iterations
        nr_bins (int): number to bin trial

    Returns:
        np ar: array of binns (i=cluster,j=bin,k=iteration, data=bin count)
    """
    cluster=input_ar.shape[0]
    iterations=input_ar.shape[2]
    # y = cluster index
    # x = bin number 1 to 50
    # z = random iteration 1 to 1000
    data_ar=np.zeros(shape=(cluster,nr_bins,iterations),dtype=int)
    for cl in range(cluster):
        for it in range(iterations):
            data_ar[cl,:,it]=(np.histogram(np.concatenate(input_ar[cl,:,it]).ravel(),bins=nr_bins))[0]
    return data_ar





# load alls session

all_sessions_dict = dict()
# load all sessions into dict
for session_name,missing_rows_ttl,load_spikes,spikes_trials_skip in se_li:
    all_sessions_dict[session_name]=load_session(session_name, missing_rows_ttl, load_spikes, spikes_trials_skip)




def convert_numeric(columns, df):
    for column in columns:
        df[column] = pd.to_numeric(df[column])
    return df

def get_trial_info(all_sessions_dict):
    session_info_df = pd.DataFrame(columns=['id','blocks','tot. trials','wheel ns trials','no resp trials','selected trials',
                                           'reward','no-reward','gamble', 'safe',
                                           'gamble rewarded','safe reward','gamble no-reward','safe no-reward',
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
        no_resp_trials = (combined['CSV Event']=='no response in time').sum(axis=0)
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
                                           'gamble rewarded','safe reward', 'gamble no-reward', 'safe no-reward'],
                       session_info_df)
        
    return session_info_df


def boxplot(df,columns,scatter=True,title=None):
    fig,ax = plt.subplots()
    session_info_df.boxplot(column=columns,return_type='axes',ax=ax)
    
    if scatter:
        i = 1
        for column in columns:
            y=session_info_df[column] 
            x=np.ones(y.shape[0])*i
            ax.plot(x, y, 'r.', alpha=0.5)
            i+=1
        ax.plot(x,y,'r.',alpha=0.5,label="session")
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.legend()

    return fig,ax


session_info_df=get_trial_info(all_sessions_dict)
session_info_df


print(session_info_df.to_latex(index=False)) 


# wheel not stop vs other
fig,ax = boxplot(session_info_df,['tot. trials','wheel ns trials','no resp trials','selected trials' ],True)
save_fig("boxplot_whell_ns", fig)

# gamble vs save
fig,ax = boxplot(session_info_df,['no resp trials','gamble','safe'],True)
save_fig("boxplot_gamble_safe", fig)

# reward vs no-reward

fig,ax = boxplot(session_info_df,['reward','no-reward'],True)
save_fig("boxplot_rw_norw", fig)


session = 'NAME'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl = []
# behavior
deselect_trials = [(X,X),(X,'end')]
# sda
window = 2000
iterations = 1000
bins = 50

#sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
#behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
#eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)

"""
NAME = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
"""





get_ipython().run_cell_magic("time", "", """%matplotlib agg
NAME.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
NAME.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


NAME.sda.info_df


get_ipython().run_cell_magic("time", "", """NAME.report.generate_report()""")


del JG14_190621


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
missing_rows = ()
combined = sync_obj.combine_dataframes(missing_rows, additonal_rows, 'channel 1', csv)

qgrid.show_grid(combined)


sync_obj.ttl_info_channel


combined['Delta (TTL-CSV)'].plot()


combined.iloc[1720:1730]


NAME.eda.plt_trial_length()


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
NAME.eda.trials_df.loc[0:1, 'select'] = False
# end
NAME.eda.trials_df.loc[215:, 'select'] = False
NAME.eda.trials_df


NAME.eda.plt_trial_hist_and_fit(
    NAME.eda.trials_df.loc[NAME.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








session = 'JG14_190619'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl = [111,2780]
# behavior
deselect_trials = []#(X,X),(X,'end')
# sda
window = 2000
iterations = 1000
bins = 50

sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
#eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)
"""
JG14_190619 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
"""





get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190619.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190619.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


JG14_190619.sda.info_df


get_ipython().run_cell_magic("time", "", """JG14_190619.report.generate_report()""")


del JG14_190619


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
rows_missing_ttl = [111,2780]
combined = sync_obj.combine_dataframes(align=True,rows_missing_ttl=False)

qgrid.show_grid(combined)


sync_obj.ttl_info_channel


plt.figure()
combined['Delta (TTL-CSV)'].plot()
plt.show()


combined.iloc[1720:1730]


deselect_trials = [(0,1),(259,'end')]
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)


get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.selected_trials_df)


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
JG14_190619.behavior.seleted_trials_df.loc[0:1, 'select'] = False
# end
JG14_190619.behavior.selected_trials_df.loc[215:, 'select'] = False
JG14_190619.behavior.selected_trials_df


JG14_190619.eda.plt_trial_hist_and_fit(
    JG14_190619.eda.trials_df.loc[JG14_190619.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








session = 'JG14_190626'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl =  [1428, 1824, 1838, 2861, 2910, 3089, 3245, 3430, 3443]
# behavior
deselect_trials = []#(X,X),(X,'end')
# sda
window = 2000
iterations = 1000
bins = 50

#sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
#behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
#eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)
"""
JG14_190619 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
"""





get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190626.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190626.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


JG14_190626.sda.info_df


get_ipython().run_cell_magic("time", "", """JG14_190626.report.generate_report()""")


del JG14_190626


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
combined = sync_obj.combine_dataframes(align=True,rows_missing_ttl=rows_missing_ttl)

qgrid.show_grid(combined)


sync_obj.ttl_info_channel


get_ipython().run_line_magic("matplotlib", " notebook")
fig,ax = plt.subplots()
combined['Delta (TTL-CSV)'].plot()
ax.set_xlabel("Event")
ax.set_ylabel("Time delta [ms]")

"""
ax[0].set_yticks(np.arange(start, stop, step).astype(int))
# set tick labels
stop = trials.index.size
label = trials.index.values[np.arange(start, stop, step).astype(int)]
label = np.append(label, trials.index.values[-1])
ax[0].set_yticklabels(label)
"""

plt.show()





combined.iloc[1720:1730]


deselect_trials = [(0,1),(276,'end')]
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)




get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.selected_trials_df)


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
JG14_190626.behavior.seleted_trials_df.loc[0:1, 'select'] = False
# end
JG14_190626.behavior.selected_trials_df.loc[215:, 'select'] = False
JG14_190626.behavior.selected_trials_df


JG14_190626.eda.plt_trial_hist_and_fit(
    JG14_190626.eda.trials_df.loc[JG14_190619.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








session = 'JG15_190722'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl =  [2094, 2574, 2637, 2808, 2831, 3499]
# behavior
deselect_trials = []#(X,X),(X,'end')
# sda
window = 2000
iterations = 1000
bins = 50

sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)

JG14_190619 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    #'sda': sda_obj,
                    #'report': report_obj,
                   })






get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


JG15_190722.sda.info_df


get_ipython().run_cell_magic("time", "", """JG15_190722.report.generate_report()""")


del JG15_190722


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
combined = sync_obj.combine_dataframes(align=False,rows_missing_ttl=rows_missing_ttl)

qgrid.show_grid(combined)



combined.index


sync_obj.ttl_info_channel


plt.figure()
combined['Delta (TTL-CSV)'].plot()
plt.show()


combined.iloc[1720:1730]


deselect_trials = [(271,'end')]
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)




get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.selected_trials_df)


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
JG15_190722.behavior.seleted_trials_df.loc[0:1, 'select'] = False
# end
JG15_190722.behavior.selected_trials_df.loc[215:, 'select'] = False
JG15_190722.behavior.selected_trials_df


JG15_190722.eda.plt_trial_hist_and_fit(
    JG15_190722.eda.trials_df.loc[JG14_190619.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








session = 'JG15_190725'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl =  [366,711,1487,1578,1659,2221,2666,2720,2769,2847,3371,3476]
# behavior
deselect_trials = []#(X,X),(X,'end')
# sda
window = 2000
iterations = 1000
bins = 50

#sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
#behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
#eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)
"""
JG14_190619 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
"""





get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


JG15_190722.sda.info_df


get_ipython().run_cell_magic("time", "", """JG15_190722.report.generate_report()""")


del JG15_190722


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
combined = sync_obj.combine_dataframes(align=True,rows_missing_ttl=rows_missing_ttl)

qgrid.show_grid(combined)


sync_obj.ttl_info_channel


plt.figure()
combined['Delta (TTL-CSV)'].plot()
plt.show()


combined.iloc[1720:1730]


deselect_trials = [(184,'end')]
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)




get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.selected_trials_df)


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
JG15_190722.behavior.seleted_trials_df.loc[0:1, 'select'] = False
# end
JG15_190722.behavior.selected_trials_df.loc[215:, 'select'] = False
JG15_190722.behavior.selected_trials_df


JG15_190722.eda.plt_trial_hist_and_fit(
    JG15_190722.eda.trials_df.loc[JG14_190619.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








session = 'JG18a_190814'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl =   [405,2621,2693,2770,2959,3015,3029,3038,3048]
# behavior
deselect_trials = []#(X,X),(X,'end')
# sda
window = 2000
iterations = 1000
bins = 50

#sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
#behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
#eda_obj = SpikesEDA(behavior_obj)
#sda_obj = SpikesSDA(eda_obj)
#report_obj = SpikesReport(eda_obj)
"""
JG14_190619 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
"""





get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.eda.generate_plots()""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG15_190722.sda.generate_plots(window,iterations,bins,individual=True,load=True)""")


JG15_190722.sda.info_df


get_ipython().run_cell_magic("time", "", """JG15_190722.report.generate_report()""")


del JG15_190722


# load session initially
sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 


ttl1 = sync_obj.ttl_signals['channel 1']
qgrid.show_grid(ttl1)


csv = sync_obj.csv
qgrid.show_grid(csv)


# first find missing rows
combined = sync_obj.combine_dataframes(align=True,rows_missing_ttl=rows_missing_ttl)

qgrid.show_grid(combined)


sync_obj.ttl_info_channel


plt.figure()
combined['Delta (TTL-CSV)'].plot()
plt.show()


combined.iloc[1720:1730]


deselect_trials = [(307,'end')]
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)




get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.selected_trials_df)


get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.good_trials_df)


trials_df=behavior_obj.good_trials_df

fig, ax = plt.subplots()
ax.plot(trials_df['length'])
ylabels=ax.get_yticklabels()
#labels=[item.get_text() for item in ylabels]
ax.set_yticklabels([1,2,3,4,5,6,7,8,9,10])
ax.set_xlabel("trial")
ax.set_ylabel("length [s]")



label=[item.get_text() for item in ylabels]
label
#label_new = np.array(labels).astype(int)


new_labels=[item.get_text() for item in labels]
new_labels


np.array(new_labels).astype(int)


/20000).astype(int).astype(str)


fig=plt.figure()
behavior_obj.good_trials_df['length'].plot.hist()


qgrid.show_grid(behavior_obj.good_trials_df.sort_values("length",ascending=False))


behavior_obj.good_trials_df[behavior_obj.good_trials_df['event']=='no response in time'].shape[0]


get_ipython().run_line_magic("matplotlib", " notebook")
behavior_obj.plt_trial_length(behavior_obj.all_trials_df)


#spikes.trials_df.loc[np.r_[0:6, 215:], 'select']
# start
JG15_190722.behavior.seleted_trials_df.loc[0:1, 'select'] = False
# end
JG15_190722.behavior.selected_trials_df.loc[215:, 'select'] = False
JG15_190722.behavior.selected_trials_df


JG15_190722.eda.plt_trial_hist_and_fit(
    JG15_190722.eda.trials_df.loc[JG14_190619.eda.trials_df.loc[:, 'select'], 'length'])


deselect_trials = [(X,X),(X,'end')]








get_ipython().run_cell_magic("time", "", """
session = 'JG14_190621'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl = [1900,1931,1996,2058,2127]
# behavior
deselect_trials = [(0,6),(215,'end')]
# sda
window = 2000
iterations = 1000
bins = 50

sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
eda_obj = SpikesEDA(behavior_obj)
sda_obj = SpikesSDA(eda_obj)
report_obj = SpikesReport(eda_obj)

JG14_190621 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })
""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190621.eda.generate_plots(window,bins)""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG14_190621.sda.generate_plots(window,iterations,bins,individual=False,reload_data_dict=True,reload_spikes_ar=False)""")


JG14_190621.sda.info_df


get_ipython().run_cell_magic("time", "", """JG14_190621.report.generate_report()""")


del JG14_190621











get_ipython().run_cell_magic("time", "", """
session = 'JG18b_190828'
folder = mac_folder + r"/" + session
# sync
rows_missing_ttl = [1744, 2363, 2648, 2701, 2731, 2778,2953,2967]
# behavior
deselect_trials = [(0,0),(204,'end')]
# eda
skip_clusters = [116, 280]
# sda
window = 2000
iterations = 1000
bins = 50

sync_obj = SyncPhenosys(session, folder, 7, 1, rows_missing_ttl) 
behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
eda_obj = SpikesEDA(behavior_obj)
sda_obj = SpikesSDA(eda_obj)
report_obj = SpikesReport(eda_obj)

JG18b_190828 = type('obj', (object,), 
                   {'sync':sync_obj,
                    'behavior':behavior_obj,
                    'eda':eda_obj,
                    'sda': sda_obj,
                    'report': report_obj,
                   })

""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG18b_190828.eda.generate_plots(window,bins)""")


get_ipython().run_cell_magic("time", "", """%matplotlib agg
JG18b_190828.sda.generate_plots(window,iterations,bins,individual=False,reload_data_dict=True,reload_spikes_ar=False)""")


JG18b_190828.sda.info_df


get_ipython().run_cell_magic("time", "", """#%matplotlib agg
JG18b_190828.report.generate_report()""")



