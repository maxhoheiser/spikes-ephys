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

from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects

import copy

default = [6.4, 4.8]

class BehaviorAnalysis():
    """[behaviour analysis class for phenosys Behavior Recording and Neuron Electrophysiology Recording]
    """    
    def __init__(self, sync_obj, deselect_trials=[]):
        """[summary]

        Args:
            session ([type]): [description]
            folder ([type]): [description]
            combined (pd): pandas datafame containing all the session information, from sync_class
        """        
        self.session = sync_obj.session
        self.folder = sync_obj.folder
        self.combined_df = sync_obj.combined_df
        self.gamble_side = sync_obj.gamble_side
        self.deselect_trials = deselect_trials

        self.all_trials_df = sync_obj.all_trials_df
        self.good_trials_df = sync_obj.good_trials_df

        # deselct not selected trials
        # reset
        self.good_trials_df['select'] = True
        for a,b in deselect_trials:
            if b == 'end':
                self.good_trials_df.loc[a:,'select'] = False
            else:
                self.good_trials_df.loc[a:b,'select'] = False

        self.selected_trials_df = self.good_trials_df.loc[self.good_trials_df['select'],:]
        self.selected_trials_df.reset_index(drop=True,inplace=True)


    def get_wheel_and_resp(self):
        wheel = []
        response = []
        working = self.combined.copy()
        working.reset_index(inplace=True)
        for index, row in working.iterrows():
            if index+4 == working.shape[0]:
                break 
            if (row['CSV Event'] == 'start') & ( (working.loc[index+1,'CSV Event'])=='wheel not stopping'):
                wheel.append(1)
                response.append(0)
            elif (row['CSV Event'] == 'start') & ( (working.loc[index+4,'CSV Event']) == 'no response in time'):
                response.append(1)
                wheel.append(0)
            elif (row['CSV Event'] == 'start') and ( ((working.loc[index+1,'CSV Event'])!='wheel not stopping') or  ((working.loc[index+4,'CSV Event']) != 'no response in time') ):
                wheel.append(0)
                response.append(0)
        
        self.behav_df = pd.DataFrame({'resp':response, 'wheel':wheel})
        # add rolling average
        resp_rol_df = pd.DataFrame(self.behav_df.resp.rolling(window=10).mean())
        resp_rol_df.fillna(0, inplace=True)
        wheel_rol_df = pd.DataFrame(self.behav_df.wheel.rolling(window=10).mean())
        wheel_rol_df.fillna(0, inplace=True)
        # add to dataframe
        self.behav_df['resp_rol']=resp_rol_df.values
        self.behav_df['wheel_rol']=wheel_rol_df.values

        return self.behav_df

    def plot_wheel_resp(self, wheel=True, resp=False, legend=False):
        if not 'behav_df' in dir(self):
            self.get_wheel_and_resp()
        plt.figure(figsize=(9, 4))
        if wheel:
            plt.plot(self.behav_df.wheel_rol, label='wheel not stopping')
        if resp:
            plt.plot(self.behav_df.resp_rol, label='no response in time')
        if legend:
            plt.legend(loc=(0.02, 0.1))
        plt.ylabel('rolling average (10)')
        plt.xlabel('trial')
        plt.show()
        
        # plot spike times
    def plt_trial_length(self, trials_df):
        fig, ax = plt.subplots()
        ax.plot(trials_df.loc[trials_df.loc[:,'select'],'length'])
        labels=ax.get_xticklabels()
        ax.set_xlbael("trial")
        ax.set_ylabel("length [ms]]")
        return fig,ax


    def convert_numeric(self, columns, df):
        for column in columns:
            df[column] = pd.to_numeric(df[column])
        return df

    def create_session_info_df(self):
        info_df = pd.DataFrame(columns=['block','tot. trials','reward','no-reward','gamble', 'safe',
                                        'gamble reward','safe reward', 'gamble no-reward', 'safe no-reward']
                            )

        session_name=self.session
        gamble_side=self.gamble_side


        # get info of all =============================================
        trials_df=self.selected_trials_df

        trials = trials_df.shape[0]
        rw=(trials_df["reward_given"]==True).sum()
        norw=(trials_df["reward_given"]==False).sum()
        right = (trials_df["right"]==True).sum()
        right_rw = (trials_df[(trials_df["right"])&(trials_df["reward_given"])]).shape[0]
        right_norw = (trials_df[(trials_df["right"])&np.invert(trials_df["reward_given"])]).shape[0]
        left = (trials_df["left"]==True).sum()
        left_rw = (trials_df[(trials_df["left"])&(trials_df["reward_given"])]).shape[0]
        left_norw = (trials_df[(trials_df["left"])&np.invert(trials_df["reward_given"])]).shape[0]

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

        info_df.loc[info_df.shape[0] + 1] = ['all', trials, rw, norw, gamble, safe,gamble_rw,
                                            safe_rw, gamble_norw, safe_norw
                                            ]

        # get info form blocks =============================================
        for block in [0.75,0.25,0.125]:
            block_df = trials_df[trials_df.probability==block]

            trials = block_df.shape[0]
            rw=(block_df["reward_given"]==True).sum()
            norw=(block_df["reward_given"]==False).sum()

            right = (block_df["right"]==True).sum()
            right_rw = (block_df[(trials_df["right"])&(block_df["reward_given"])]).shape[0]
            right_norw = (block_df[(block_df["right"])&np.invert(block_df["reward_given"])]).shape[0]
            left = (block_df["left"]==True).sum()
            left_rw = (block_df[(block_df["left"])&(block_df["reward_given"])]).shape[0]
            left_norw = (block_df[(block_df["left"])&np.invert(block_df["reward_given"])]).shape[0]

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

            info_df.loc[info_df.shape[0] + 1] = [block, trials, rw, norw, gamble, safe,gamble_rw,
                                                safe_rw, gamble_norw, safe_norw
                                                ]


        info_df = self.convert_numeric(['tot. trials','reward','no-reward','gamble', 'safe',
                                'gamble reward','safe reward', 'gamble no-reward', 'safe no-reward'],
                                info_df
                                )

        self.info_df = info_df
        return info_df

    def colormap(self, info_df,figsize=default):
        x_labels = info_df.columns.values[1:].tolist()
        y_labels = ['all trials','75% block', "25% block", "12.5% block"]


        data = info_df.loc[:,['tot. trials','reward','no-reward','gamble', 'safe','gamble reward','safe reward', 'gamble no-reward', 'safe no-reward']].values

        fig, ax = plt.subplots(1,1,figsize=figsize)
        #im = ax.imshow(data,norm=LogNorm())

        my_cmap = copy.copy(mpl.cm.get_cmap('viridis')) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        im = ax.imshow(data, 
                norm=mpl.colors.LogNorm(), 
                interpolation='nearest', 
                cmap=my_cmap,
                rasterized=True)

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
                            ha="center", va="center", color="w", fontsize=14)
                text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                path_effects.Normal()])

        ax.grid(False)

        #ax.set_title("Distribution of Trials")
        #fig.colorbar(im)
        fig.tight_layout()
        return fig,ax
