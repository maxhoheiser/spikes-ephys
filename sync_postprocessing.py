"""
open ephys sync recording postprocessing  functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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

import seaborn as sns

from os import path
import os
import sys
import platform

default = (6,4)

def load_oo_events(rec_folder):
    #define event folde
    if platform.system() == 'Linux':
        event_folder = rec_folder+r"/events/Rhythm_FPGA-100.0/TTL_1"
    elif platform.system() == 'Windows':
        event_folder = rec_folder+r"\events\Rhythm_FPGA-100.0\TTL_1"
    elif platform.system() == 'Darwin': #macos
        event_folder = rec_folder+r"/events/Rhythm_FPGA-100.0/TTL_1"
    
    # load infos
    channel_states_ar = np.load(path.join(event_folder,'channel_states.npy'))
    channels_ar = np.load(path.join(event_folder,'channels.npy'))
    #text_ar = np.load(path.join(event_folder,'text.npy'))
    full_words = np.load(path.join(event_folder,'full_words.npy'))
    timestamps_ar = np.load(path.join(event_folder,'timestamps.npy'))
    # normalize
    timestamps_nor_ar = timestamps_ar - timestamps_ar[0]
    
    # create array
    oe_events_ar = np.zeros([timestamps_nor_ar.shape[0],5])
    oe_events_ar[:,0]=timestamps_ar
    oe_events_ar[:,1]=timestamps_nor_ar/30
    oe_events_ar[:,2]=timestamps_nor_ar/30000
    oe_events_ar[:,3]=channel_states_ar
    oe_events_ar[:,4]=channels_ar
    
    # create dataframe
    oe_events_df = pd.DataFrame(oe_events_ar,columns=["samplerate_absolut","ms_relativ","s_relativ","event","channel"])
    return oe_events_df

def extract_events_oo(oe_events_df):
    oe_sync_df = (oe_events_df.loc[np.logical_or(oe_events_df['channel']==1, oe_events_df['channel']==2)]).copy()
    oe_sync_df.reset_index(inplace=True,drop=True)
    oe_trials = []

    trial=False
    bnc1=-1
    bnc2=-1
    for row in  oe_sync_df.index.values[1:]:
        # check bnc state
        if oe_sync_df.loc[row,"channel"]==1:
            bnc1=oe_sync_df.loc[row,"event"]
        if oe_sync_df.loc[row,"channel"]==2:
            bnc2=oe_sync_df.loc[row,"event"]
        # 1 2 = start
        if oe_sync_df.loc[row,"event"]==2 and oe_sync_df.loc[row-1,"event"]==1:
            if oe_sync_df.loc[row,"ms_relativ"]==oe_sync_df.loc[row-1,"ms_relativ"]:
                if trial == False:
                    oe_trials.append([oe_sync_df.loc[row,"ms_relativ"],"start"])
                    #oe_trials.append([oe_sync_df.loc[row+1,"ms_relativ"],"event"])
                    trial=True
                elif trial == True:
                    oe_trials.append([oe_sync_df.loc[row,"ms_relativ"],"event"])
                    oe_trials.append([oe_sync_df.loc[row+1,"ms_relativ"],"event"])
                    oe_trials.append([oe_sync_df.loc[row+1,"ms_relativ"],"end"])
                    trial=False

        # 1 0 = reward event
        if oe_sync_df.loc[row,"channel"]==1 and trial:
            if not(oe_sync_df.loc[row+1,"channel"]==2 and oe_sync_df.loc[row,"ms_relativ"]==oe_sync_df.loc[row+1,"ms_relativ"]):
                oe_trials.append([oe_sync_df.loc[row,"ms_relativ"],"reward_event"])
                #oe_events.append([oe_sync_df.loc[row+1,"s_relativ"],"reward_end"])

        # 0 1 = normal event
        if oe_sync_df.loc[row,"channel"]==2:
            if not(oe_sync_df.loc[row-1,"event"]==1 and oe_sync_df.loc[row,"ms_relativ"]==oe_sync_df.loc[row-1,"ms_relativ"]) and trial:
                oe_trials.append([oe_sync_df.loc[row,"ms_relativ"],"event"])
                #oe_events.append([oe_sync_df.loc[row+1,"s_relativ"],"reward_end"])


    oe_trials_df = pd.DataFrame(oe_trials,columns=["ms_relativ","event_type"]) 
    return oe_trials_df

def convert_to_seconds(csv_string):
    utc_time = datetime.strptime(csv_string,
                                "%Y-%m-%d %H:%M:%S.%f"
                                )
    return utc_time.timestamp()

def load_bp_events(root_dir,session):
    #specify path
    if platform.system() == 'Linux':
        event_folder = (root_dir + "/experiments/gamble_task/setups/gamble_task_recording/sessions")
    elif platform.system() == 'Windows':
        folder = (root_dir + r"\experiments\gamble_task\setups\gamble_task_recording\sessions")
    elif platform.system() == 'Darwin': #macos
        folder= (root_dir + "/experiments/gamble_task/setups/gamble_task_recording/sessions")
    
    ext = ".csv"
    # read csv
    session_df = pd.read_csv(path.join(folder,session,session)+ext,sep=';',header=6)
    #convert string to datetime
    session_df["datetime"]=(pd.to_datetime(session_df["PC-TIME"].values,format="%Y-%m-%d %H:%M:%S.%f")).values
    # get milliseconds
    session_df["ms_absolut"]=session_df["datetime"].apply(lambda x: x.timestamp()*1000)
    session_df["ms_relativ"]=session_df["ms_absolut"]-session_df.loc[14,"ms_absolut"]

    return session_df

def get_sync(openephys_dir,pybpod_root,pybpod_session):
    pb_events_df = load_bp_events(pybpod_root,pybpod_session)
    # extract states
    states_df = pb_events_df.loc[pb_events_df.TYPE=='STATE']
    states_df = states_df.dropna(axis=0,how='any')

    starts = pb_events_df.loc[pb_events_df.loc[(pb_events_df.TYPE=='TRIAL')].index,'ms_relativ'].copy()
    starts.reset_index(inplace=True,drop=True)

    start_idx = -1
    trial=False
    for idx in states_df.index:
        if states_df.loc[idx,'BPOD-INITIAL-TIME']==0:
            start_idx+=1
        states_df.loc[idx,'ms_relativ']=states_df.loc[idx,'BPOD-INITIAL-TIME']*1000+starts.loc[start_idx]

    pb_sync_df = states_df#states_df.loc[:,["MSG","ms_relativ","BPOD-INITIAL-TIME"]]


    # load openephys ttl
    oe_events_df = load_oo_events(openephys_dir)
    # conver ttl to events
    oe_trials_df = extract_events_oo(oe_events_df)
    # remove end
    oe_end_idx = oe_trials_df.loc[oe_trials_df.event_type=='end'].index
    not_select = oe_trials_df.index.isin(oe_end_idx.values-1)
    oe_trials_df = oe_trials_df.loc[~not_select]

    #oe_trials_df=oe_trials_df.loc[np.invert(oe_trials_df.event_type=='end')]


    """old
    # create combined df
    # select bpod necessary files
    selector = np.logical_or(np.logical_or(pb_events_df.TYPE=="TRANSITION",pb_events_df.TYPE=="TRIAL"),pb_events_df.TYPE=="END-TRIAL")
    pb_sync_df = pb_events_df.loc[selector,["MSG","ms_relativ","BPOD-INITIAL-TIME"]]
    # combine both
    combined_ar = np.zeros([pb_sync_df.shape[0],4],dtype=object)
    combined_ar[:oe_trials_df.shape[0],[0,2]]=oe_trials_df.values
    combined_ar[:pb_sync_df.shape[0],[1,3]]=pb_sync_df.loc[:,["ms_relativ","MSG"]].values
    combined_df=pd.DataFrame(combined_ar,columns=["oe_ms_relativ","pb_ms_relativ","oe_event","pb_event"])
    combined_df["delta_pb_oe"]=combined_df["pb_ms_relativ"]-combined_df["oe_ms_relativ"]
    """


    # create combined
    combined_ar = np.zeros([pb_sync_df.shape[0],11],dtype=object)
    combined_ar[:oe_trials_df.shape[0],0:2]=oe_trials_df.values
    combined_ar[:pb_sync_df.shape[0],2:11]=pb_sync_df.values
    combined_df=pd.DataFrame(combined_ar,columns=["TTL Start norm","TTL Event","CSV Type","CSV Pctime","CSV in trial start","CSV in trial end",
                                                "CSV Event","CSV info","CSV Datetime","CSV Start","CSV Start norm"])
    combined_df["Delta (TTL-CSV)"]=combined_df["TTL Start norm"]-combined_df["CSV Start norm"]

    # model like phenosys sync df for futher analysis
    combined_df.reset_index(inplace=True,drop=True)

    return combined_df





# Plotting =============================================================================================

def plt_start_stop_dif(combined_df,figsize=default):
    # get data
    start=(combined_df.loc[combined_df['CSV Event']=='start']).copy()
    start.reset_index(inplace=True,drop=True)
    stop=(combined_df.loc[combined_df['CSV Event']=='end_state']).copy()
    stop.reset_index(inplace=True,drop=True)

    # plot data
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(start['Delta (TTL-CSV)'],color='r',label='start')
    ax.plot(stop['Delta (TTL-CSV)'],color='g',label='end')

    # labeling
    ax.set_xlabel("Trial")
    ax.set_ylabel("Time difference [ms]")
    #ax.grid()
    ax.legend()
    
    return fig,ax


def plt_event_dif(combined_df,plot_all=False,figsize=(12,5.5)):
    data = combined_df.loc[:,["TTL Start norm","TTL Event", "CSV Event","CSV Start norm","Delta (TTL-CSV)"]].copy()
    data.columns=['oe_ms_relativ','oe_event','pb_event','pb_ms_relativ','delta_pb_oe']

    if not(plot_all):
        not_list = ['wheel_stopping_check',
                    'wheel_stopping_check_failed_punish',
                    'wheel_stopping_check_failed_reset',
                    'reset_rotary_encoder_wheel_stopping_check',
                    'wheel not stopping'
                    ]
        bool_filter = ~data.pb_event.isin(not_list)
        data = data.loc[bool_filter].copy()
        data.reset_index(inplace=True,drop=True)
        data.delta_pb_oe = data.delta_pb_oe.astype(float)
    else:
        data.delta_pb_oe = data.delta_pb_oe.astype(float)

    new_trial = data.loc[data.pb_event=='start']

    fig,ax = plt.subplots(figsize=figsize)
    # scatterplot with color
    g = sns.scatterplot(data=data, x=data.index, y='delta_pb_oe', hue='pb_event', s=70)
    # connect dots
    sns.lineplot(data=data.delta_pb_oe)
    # add trial start lines
    line_min , line_max = ax.get_ylim()#data.delta_pb_oe.max()
    #line_min = #data.delta_pb_oe.min()
    plt.vlines(new_trial.index.values, line_min, line_max, linestyle='-',alpha=0.1,color='k',linewidth=1.2) #,label='New trial'
    # trial 2nd x axis
    trial=1
    for trial_idx in new_trial.iterrows():
        ax.text(trial_idx[0]-0.3, line_max+(line_max/50), f"{trial}")
        trial+=1

    # move index outside
    g.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    ax.set_ylabel('Time Delta [ms]')
    ax.set_xlabel('Event')
    # second x label
    fig.text(0.5,0.93,"Trial")

    # set y lims
    ax.set_ylim(line_min,line_max)

    return fig,ax