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




    # Behavior Plot Gauss Smoothed ============================================================================================================

    def get_behavior_df(self,fwhm=4):
        working = self.all_trials_df.copy()
        working.reset_index(inplace=True)
        if self.gamble_side=='right':
            gamble_side = 'right'
            safe_side = 'left'
        else:
            gamble_side = 'left'
            safe_side = 'right'
            
        behav_df = pd.DataFrame({'wheel':(working['event']=='wheel not stopping').astype(int), 
                                'resp':(working['event']=='no response in time').astype(int),
                                'gamble_rw':(working['event']==f"{gamble_side}_rw").astype(int),
                                'gamble_norw':(working['event']==f"{gamble_side}_norw").astype(int),
                                'safe_rw':(working['event']==f"{safe_side}_rw").astype(int),
                                'safe_norw':(working['event']==f"{safe_side}_norw").astype(int),
                                })

        # get averaged occurance 
        resp_rol = pd.DataFrame(behav_df['resp'].rolling(window=10).mean())
        resp_rol.fillna(0, inplace=True)
        behav_df[f"{'resp'}_rol"]=resp_rol
        wheel_rol = pd.DataFrame(behav_df['wheel'].rolling(window=10).mean())
        wheel_rol.fillna(0, inplace=True)
        behav_df[f"{'wheel'}_rol"]=wheel_rol
        # gamble
        gamble_rw_rol = pd.DataFrame(behav_df['gamble_rw'].rolling(window=10).mean())
        gamble_rw_rol.fillna(0, inplace=True)
        gamble_norw_rol = pd.DataFrame(behav_df['gamble_norw'].rolling(window=10).mean())
        gamble_norw_rol.fillna(0, inplace=True)
        safe_rw_rol = pd.DataFrame(behav_df['safe_rw'].rolling(window=10).mean())
        safe_rw_rol.fillna(0, inplace=True)
        safe_norw_rol = pd.DataFrame(behav_df['safe_norw'].rolling(window=10).mean())
        safe_norw_rol.fillna(0, inplace=True)
        # normalize rol
        gamble_max = (np.concatenate((gamble_rw_rol.values,gamble_norw_rol.values))).max()
        gamble_rw_rol = gamble_rw_rol/gamble_max
        gamble_norw_rol = gamble_norw_rol/gamble_max
        #
        safe_max = (np.concatenate((safe_rw_rol.values,safe_norw_rol.values))).max()
        safe_rw_rol = safe_rw_rol/safe_max
        safe_norw_rol = safe_norw_rol/safe_max
        # add to df
        behav_df[f"{'gamble_rw'}_rol"]=gamble_rw_rol
        behav_df[f"{'gamble_norw'}_rol"]=gamble_norw_rol
        behav_df[f"{'safe_rw'}_rol"]=safe_rw_rol
        behav_df[f"{'safe_norw'}_rol"]=safe_norw_rol
        
        # get gaus smoothed
        for column in ['resp_rol','wheel_rol','gamble_rw_rol','gamble_norw_rol','safe_rw_rol','safe_norw_rol']:
            gaus = self.gauss_smooth(behav_df[column],fwhm=fwhm)
            behav_df[f"{column[:-4]}_gaus"]=gaus

        self.behavior_df=behav_df
        return self.behavior_df

    def fwhm2sigma(self,fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    def gauss_smooth(self,data,fwhm=4):
        sigma = self.fwhm2sigma(fwhm)

        x_vals= data.index.values
        y_vals = np.ravel(data.values)

        smoothed_vals = np.zeros(y_vals.shape)
        for x_position in x_vals:
            kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
            kernel = kernel / sum(kernel)
            smoothed_vals[x_position] = sum(y_vals * kernel)
            
        return smoothed_vals



    def plot_wheel_resp(self, wheel=True, resp=False, legend=False, gauss=True, unique=True, fwhm=4, figsize=(9,6),loc=(0.015, 0.1)):
        #if not 'behavior_df' in dir(self):
        self.get_behavior_df(fwhm=fwhm)

        fig,ax  = plt.subplots(1,1,figsize=figsize)
        if gauss:
            if wheel:
                ax.plot(self.behavior_df.wheel_gaus, label='wheel not stp.',color=blue)
            if resp:
                ax.plot(self.behavior_df.resp_gaus, label='no resp.',color='k')
        else:
            if wheel:
                ax.plot(self.behavior_df.wheel_rol, label='wheel not stp.',color=blue)
            if resp:
                ax.plot(self.behavior_df.resp_rol, label='no resp.',color='k')
        if unique:
            resp_uniqe_x=list()
            wheel_uniqe_x=list()

            for x,y in self.behavior_df.iterrows():
                if y.resp != 0:
                    resp_uniqe_x.append(x)
                if y.wheel != 0:
                    wheel_uniqe_x.append(x)
            # plot unique lines
            # wheel
            ax.eventplot(wheel_uniqe_x,lineoffsets=1.1,linelengths=0.1,color=blue,linewidth=0.5)  #y_min=1.1,y_max=1.2
            # resp
            ax.eventplot(resp_uniqe_x,lineoffsets=-0.1,linelengths=0.1,color='k',linewidth=0.5) #y_min=-1.2,y_max=-1.1

        # plot prob block changes
        blocks = self.all_trials_df['probability'].unique()
        for block in blocks:
            occurance=(np.where(self.all_trials_df['probability']==block)[0][0])
            ax.axvline(occurance,0,1,linestyle='-',color='k',linewidth=0.5,alpha=0.4)
            ax.text(occurance-15, 1.35, f"{block*100}%")

        if legend:
            ax.legend(loc=loc)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Trial')
        # secondary axis
        fig.text(0.48,1.02,"Block")

        return fig,ax


    def plt_side_behav(self,
                        side,
                        unique=True, 
                        fwhm=4, 
                        figsize=(9,6),
                        legend=True
                        ):
        #if not 'behavior_df' in dir(self):
        self.get_behavior_df(fwhm=fwhm)

        fig,ax  = plt.subplots(1,1,figsize=figsize)

        # plot gauss smooth propability
        ax.plot(self.behavior_df[f"{side}_rw_gaus"], label='reward',color=green)
        ax.plot(self.behavior_df[f"{side}_norw_gaus"], label='no-reward',color=red)

        # plot unique lines
        if unique:
            rw=list()
            norw=list()

            for x,y in self.behavior_df.iterrows():
                if y[f"{side}_rw"] != 0:
                    rw.append(x)
                if y[f"{side}_norw"] != 0:
                    norw.append(x)
            ax.eventplot(rw,lineoffsets=1.1,linelengths=0.1,color=green,linewidth=0.8)
            ax.eventplot(norw,lineoffsets=-0.1,linelengths=0.1,color=red,linewidth=0.8)
            
        # plot prob block changes
        blocks = self.all_trials_df['probability'].unique()
        for block in blocks:
            occurance=(np.where(self.all_trials_df['probability']==block)[0][0])
            ax.axvline(occurance,0,1,linestyle='-',color='k',linewidth=0.5,alpha=0.4)
            ax.text(occurance-15, 1.35, f"{block*100}%")

        #ax.set_xlim=(-0.3,1.5)

        if legend:
            plt.legend(loc=1)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Trial')
        # secondary axis
        fig.text(0.48,1.05,"Block")

        return fig,ax

    def plt_side_compare(self,
                        reward,
                        unique=True, 
                        fwhm=4, 
                        figsize=(9,6),
                        legend=True
                        ):
        #if not 'behavior_df' in dir(self):
        self.get_behavior_df(fwhm=fwhm)

        fig,ax  = plt.subplots(1,1,figsize=figsize)

        # plot gauss smooth propability
        if reward:
            key = 'rw'
            label = 'reward'
        else:
            key = 'norw'
            label = 'no-reward'

        ax.plot(self.behavior_df[f"gamble_{key}_gaus"], label=f"gamble {label}",color=purple)
        ax.plot(self.behavior_df[f"safe_{key}_gaus"], label=f"safe {label}",color=lightblue)

        # plot unique lines
        if unique:
            rw=list()
            norw=list()

            for x,y in self.behavior_df.iterrows():
                if y[f"gamble_{key}"] != 0:
                    rw.append(x)
                if y[f"safe_{key}"] != 0:
                    norw.append(x)
            ax.eventplot(rw,lineoffsets=1.1,linelengths=0.1,color=purple,linewidth=0.8)
            ax.eventplot(norw,lineoffsets=-0.1,linelengths=0.1,color=lightblue,linewidth=0.8)
            
        # plot prob block changes
        blocks = self.all_trials_df['probability'].unique()
        for block in blocks:
            occurance=(np.where(self.all_trials_df['probability']==block)[0][0])
            ax.axvline(occurance,0,1,linestyle='-',color='k',linewidth=0.5,alpha=0.4)
            ax.text(occurance-15, 1.35, f"{block*100}%")

        #ax.set_xlim=(-0.3,1.5)

        if legend:
            plt.legend(loc=1)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Trial')
        # secondary axis
        fig.text(0.48,1.02,"Block")

        return fig,ax
        
        # plot spike times
    def plt_trial_length(self, trials_df):
        fig, ax = plt.subplots()
        ax.plot(trials_df.loc[trials_df.loc[:,'select'],'length'])
        labels=ax.get_xticklabels()
        ax.set_xlbael("Trial")
        ax.set_ylabel("Length [ms]]")
        return fig,ax


    # Plot response Time =================================================================================================================
    def plt_resp_time(self,figsize=(9,6),legend=True):
        #if not 'behavior_df' in dir(self):
        if self.gamble_side == 'right':
            gamble='right'
            safe='left'
        else:
            gamble='left'
            safe='right'
            
        # gamble times    
        df = self.all_trials_df.loc[self.all_trials_df[gamble]]
        gamble_times = (df['reward']-df['openloop'])/20000
        # safe times
        df = self.all_trials_df.loc[self.all_trials_df[safe]]
        safe_times = (df['reward']-df['openloop'])/20000


        fig,ax  = plt.subplots(1,1,figsize=figsize)
        #fig,ax = plt.subplots(1,1,figsize=set_size(0.8,size='xlong'))

        # plot response time
        ax.scatter(gamble_times.index.values,gamble_times.values,
                facecolors='none', edgecolors=purple,marker='o',linewidths=1.5, s=30, 
                label='gamble side'
                )
        ax.scatter(safe_times.index.values,safe_times.values,
                facecolors='none', edgecolors=lightblue, marker='o',linewidths=1.5,  s=30, alpha=0.8, 
                label='safe side'
                )

        ax.set_ylim([ax.get_ylim()[0],10])

        # plot prob block changes
        blocks = self.all_trials_df['probability'].unique()
        for block in blocks:
            occurance=(np.where(self.all_trials_df['probability']==block)[0][0])
            ax.axvline(occurance,0,1,linestyle='-',color='k',linewidth=0.5,alpha=0.4)
            ax.text(occurance-15, 10.6, f"{block*100}%")

        if legend:
            plt.legend(loc=2)
        ax.set_ylabel('Time [s]')
        ax.set_xlabel('Trial')
        # secondary axis
        fig.text(0.48,1.04,"Block")

        return fig,ax




    # Colormap Plots =================================================================================================================

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
