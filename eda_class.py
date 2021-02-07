# To do - test plotting of all fiugres
# pass in fig, ax to save fig function


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
from numba.experimental import jitclass
spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]
"""



# class ###################################################################################################################
#@jitclass(spec)
class SpikesEDA():
    def __init__(self, behavior_obj, skip_clusters=[]):
        self.session = behavior_obj.session
        self.folder = behavior_obj.folder
        self.gamble_side = behavior_obj.gamble_side

        self.all_trials_df = behavior_obj.all_trials_df
        self.good_trials_df = behavior_obj.good_trials_df
        self.selected_trials_df = behavior_obj.selected_trials_df
        self.skip_clusters = skip_clusters
        self.spikes_df, self.clusters_df = self.load_files()
        
        self.spikes_per_trial_ar = self.gen_spike_per_trial_matrix()
        self.spikes_per_cluster_ar = self.gen_spike_per_cluster_matrix()
        #self.randomized_bins_ar = self.get_randomized_samples(200, 1000)

    # load files from kilosort & behavior files
    def load_files(self):
        """

        """
        if platform.system() == 'Linux':
            # load fies in liux
            spike_times = np.load(self.folder+r"/electrophysiology/spike_times.npy")
            spike_cluster = np.load(self.folder+r"/electrophysiology/spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"/electrophysiology/cluster_info.tsv", sep='\t')
            #excel_df = pd.read_excel(self.folder+"/output_file.xlsx", 'Daten', header=[0, 1] )

        elif platform.system() == 'Windows':
            # load files in Windows
            spike_times = np.load(self.folder+r"\electrophysiology\spike_times.npy")
            spike_cluster = np.load(self.folder+r"\electrophysiology\spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"\electrophysiology\cluster_info.tsv", sep='\t')
            #excel_df = pd.read_excel(self.folder+r"\output_file.xlsx", 'Daten', header=[0, 1] )

        elif platform.system() == 'Darwin':
            # load fies in liux
            spike_times = np.load(self.folder+r"/electrophysiology/spike_times.npy")
            spike_cluster = np.load(self.folder+r"/electrophysiology/spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"/electrophysiology/cluster_info.tsv", sep='\t')
            #excel_df = pd.read_excel(self.folder+"/output_file.xlsx", 'Daten', header=[0, 1] )

        # create spike Data Frame with clusters and spike times
        spikes_df = pd.DataFrame( { 'cluster':spike_cluster, 'spike_times': spike_times[:,0] } )
        spikes_df.index.name = 'global index'
        # set indexes for each clusters
        spikes_df = spikes_df.set_index((spikes_df.groupby('cluster').cumcount()).rename('cluster index'), append=True)
        # clean up cluster data frame
        clusters_df = clusters_df.rename(columns={'id':'cluster id'})
        #clusters_df = clusters_df.sort_values(by=['group']).reset_index(drop=True, )
        clusters_df = clusters_df.set_index('cluster id')
        # create 'spikes' colum with spiketimes
        spk = pd.DataFrame( {'spikes':np.zeros(clusters_df.shape[0], dtype=object)}, index=clusters_df.index )
        for group, frame in spikes_df.groupby('cluster'):
            spk['spikes'][group] = frame['spike_times'].values
        #merge spike column with clusters_df
        clusters_df = pd.merge(clusters_df, spk, how='right', left_index=True, right_index=True)
        for skip in self.skip_clusters:
            clusters_df.loc[skip,'group']='mua'

        """# clean up trials data Frame
        # drop NaN from loaded excel
        excel_df.dropna(axis=0, how='all', inplace=True)
        # drop 0 leading rows from loaded excel
        excel_df = excel_df.loc[(excel_df.iloc[:,[1,2,3,4,5,6,7,8]]!=0).sum(axis=1)==8, :]
        # create cleaned up data frame with each trial in one row and times and behavior
        trials_df = excel_df.loc[:]['TTL']
        # set trials ans index and name as trials
        trials_df = trials_df.set_index('trial-num')
        trials_df.index.name = 'trial'
        # rename colums to aprop names
        trials_df = trials_df.rename(columns={'reward':'event','time 1':'start', 'time 2':'cue', 'time 3':'sound','time 4':'openl','time reward':'reward','time inter trial in.':'iti','time inter trial end':'end', 'time dif trial':'length_ms', 'ttl start rel':'rel'})
        # drop all unnecessary colums
        trials_df = trials_df.drop(['dif ttl - excel', 'diff round', 'excel start rel', 'start rel dif', 'TIstarts','IND-CUE_pres_start','SOUND_start', 'resp-time-window_start', 'ITIstarts','ITIends', 'time dif trial round', 'rel'], axis = 1 )
        # drop al rows with only 0
        trials_df = trials_df.drop(trials_df[trials_df['start']==0].index, axis=0)
        # convert times in ms to count 20k per second (*20)
        trials_df.loc[:,'start':'end']*=20
        # convert all time columns to int64
        trials_df = trials_df.astype({'start': int, 'cue': int, 'sound': int, 'openl': int, 'reward': int, 'iti': int, 'end': int})
        # calculate trial length in clicks
        trials_df['length']=trials_df['end']-trials_df['start']
        trials_df['select']=np.full((trials_df.shape[0] ,1), True ,dtype='bool')"""
        return (spikes_df, clusters_df)


##EDA#########################################################################################################################

 #Helper Functions EDA =======================================================================================================
    # find spikes between
    def get_spikes_for_trial(self, array, start, stop): #old
        '''
        params: array = numpy array (N,1) with values to check against
                start, stop = value to find all values in array between
        return: valus in array between start and stop
        '''
        ar = array[np.logical_and(array >= start, array <= stop)]
        if ar.size > 0:
            # align with start of trial
            ar = ar[:] - start
        return ar 

    # get all spikes for specified clusters
    def get_spikes_for_cluster(self, trials_df, cluster):
        '''
        params: trials_df = array with all trials, start and stop times
                sikes_times = df with all the spike times indext by cluster
                cluster = integer of cluster
        return: DataFrame with all spikes
        '''
        df = pd.DataFrame(index=[0])
        for row in trials_df.index[trials_df['select'] == True]:
            # create empty data frame indext by trials, but only one which have select = true
            start = trials_df.loc[row, 'start']
            stop = trials_df.loc[row, 'end']
            df1 = pd.DataFrame({row:self.get_spikes_for_trial(cluster, start, stop)}, dtype="Int64")
            df = pd.concat([df,df1.dropna()], axis=1)
        df = df.T
        df.index.name = 'Trial'
        return df

    # get spike frequency for cluster
    def bin_count_per_cluster(self, window, cluster, step=None):
        """
        calculate sliding bin count of spikes in bins for given bin length in ms
        cluster
        ste step between two bins 
            if none -> sliding bin with step 1
            if 1/2 window ->   1/2 overlap of each sliding bin window
        """
        bwidth_cl = window*20
        cluster = self.spikes_df.loc[self.spikes_df['cluster']==cluster]['spike_times']
        start = cluster.iloc[0].astype(int)
        end = cluster.iloc[-1].astype(int)
        if step == None:
            step = bwidth_cl+1
        else:
            step = step*20000
        # calculate
        # start of each bin
        bin_starts = np.arange(start, end+1-bwidth_cl, step)
        # end of each bin
        bin_ends = bin_starts + bwidth_cl
        # calculate index of last spike for each bin end
        last_idx = cluster.searchsorted(bin_ends, side='left').astype(int)
        # calculate index of first spike for each bin start
        first_idx = cluster.searchsorted(bin_starts, side='left')
        # return number of indexes in between start and end = number of spikes in between
        df = pd.DataFrame({'count':(last_idx - first_idx), 'start index':first_idx, 'bin end time':bin_ends ,'last spike in bin':cluster.iloc[last_idx-1].values})
        df.index.name = 'bin'
        # add trial indexes
        bins = self.selected_trials_df['end'].values
        bins = np.insert(bins, 0, 0)
        # labels
        labels = self.selected_trials_df.index.values
        # add trial index
        df['trial'] = pd.cut(df['bin end time'], bins, labels=labels, right=True, include_lowest=True)
        df.set_index('trial', append=True, inplace=True)
        df = df.swaplevel(0, 1)
        return df

    #  Compute a vector of ISIs for a single neuron given spike times.
    def compute_single_neuron_isis(self, spike_times, neuron_idx):
        """
        Compute a vector of ISIs for a single neuron given spike times.
        Args:
            spike_times (list of 1D arrays): Spike time dataset, with the first
            dimension corresponding to different neurons.
            neuron_idx (int): Index of the unit to compute ISIs for.

        Returns:
            isis (1D array): Duration of time between each spike from one neuron.
        """
        # Extract the spike times for the specified neuron
        single_neuron_spikes = spike_times.loc[neuron_idx]['spikes']

        # Compute the ISIs for this set of spikes
        # Hint: the function np.diff computes discrete differences along an array
        isis = np.diff(single_neuron_spikes)

        return isis

    # generate spike matrix
    def gen_spike_per_trial_matrix(self):
        """numpy array with all spikes for all good clusters, and all selected trials (good cluster, selected trial)

        Returns:
        array: rows=clusters, colums=trials, 
                         elements=spike times for each trial/cluster
                            spike times are aligned to each trial 0=start of trial  
                all times are in sampling points -> 20.000 spl per 1second 
        """
        trials_df = self.selected_trials_df
        spikes_df = self.spikes_df.groupby('cluster')

        all_li = []
        for group, frame in spikes_df:
            cluster_label = self.clusters_df.loc[group ,'group']
            if cluster_label == 'good':
                current_li = []
                spike_times = frame['spike_times']
                
                for row in trials_df.index:
                    start = trials_df.loc[row, 'start']
                    stop = trials_df.loc[row, 'end']
                    # get all spikes that are in the trial between start and stop + align with start of trial
                    # = spike times - trial start
                    current_li.append(self.get_spikes_for_trial(spike_times, start, stop))

                cluster_ar = np.array(current_li, dtype='object')
                all_li.append(cluster_ar)
            
        all_ar = np.array(all_li, dtype='object')
        return all_ar

    def gen_spike_per_cluster_matrix(self):
        all_ar = self.clusters_df.loc[self.clusters_df['group']=='good','spikes'].values
        return all_ar

    def get_cluster_name_from_neuron_idx(self, neuron_idx):
        cluster_name = self.clusters_df.loc[self.clusters_df['group']=='good'].iloc[neuron_idx].name
        return cluster_name

    def get_neuron_idx_from_cluster_name(self, cluster_name):
        """return the index of cluster name in only good neurons -> find in spikes_per_trial_ar
        Args:
            cluster_name (int): original index of good cluster in clusters_df 
        Returns:
            int: index of cluster in spikes_per_trial_ar
        """
        neuron_idx = (np.where(self.clusters_df.loc[self.clusters_df['group']=='good'].index.values==cluster_name))[0][0]
        return neuron_idx

 #Plotting ===================================================================================================================

    # plot histogram of trial times & normal fitted cuve
    def plt_trial_hist_and_fit(self, df, bins):
        fig, ax = plt.subplots()
        # plot histogramm
        n, bins, patches = ax.hist(df, bins, density=1)
        # add a 'best fit' line
        mean = df.mean()
        std = df.std()
        y = st.norm.pdf(bins, df.mean(), df.std())
        ax.plot(bins, y, '-')
        ax.axvline(x=mean, color='y')
        ax.set_xlabel('Trial Length [ms]')
        ax.set_ylabel('Probability density')
        #ax.set_title(f"Histogram of Trial Length $\mu=${round(mean, 2)}, $\sigma=${round(std, 4)}")
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        return fig, ax

    # plot spike times
    def plt_trial_length(self, trials_df):
        fig, ax = plt.subplots()
        ax.plot(trials_df.loc[trials_df.loc[:,'select'],'length'])

    # plot all spikes histogram
    def plt_all_cluster_spikes_hist_absolt(self):
        fig, ax = plt.subplots()

        (self.clusters_df.loc[self.clusters_df['group']=='good','n_spikes']).hist(alpha=0.6,label='number of spikes for good clusters')
        (self.clusters_df.loc[self.clusters_df['group']=='mua','n_spikes']).hist(alpha=0.4,label='number of spikes for MUA clusters')
        (self.clusters_df.loc[self.clusters_df['group']=='noise','n_spikes']).hist(label='number of spikes for noise clusters')

        ax.legend()

        ax.legend()
        ax.set_xlabel('number of spikes')
        ax.set_ylabel('cum count')
        #ax.set_title('number of spikes for specific clusters')
        
        return fig, ax

    def plt_all_cluster_spikes_hist(self):
        fig, ax = plt.subplots()

        trial_length_samplingrate = self.all_trials_df["end"].max()
        trial_length_seconds = trial_length_samplingrate / 20000

        data = self.clusters_df[['group','n_spikes']].copy()
        data['frequency'] = data['n_spikes']/trial_length_seconds

        (data.loc[data['group']=='good','frequency']).hist(alpha=0.6,label='good clusters')
        (data.loc[data['group']=='mua','frequency']).hist(alpha=0.4,label='MUA clusters')
        (data.loc[data['group']=='noise','frequency']).hist(label='noise clusters')

        ax.legend()

        ax.legend()
        ax.set_xlabel('distribution of spikes frequency across cluster types')
        ax.set_ylabel('cum count')
        #ax.set_title('number of spikes for specific clusters')
        
        return fig, ax

    # plot inter spike interval
    def plot_single_neuron_isis(self, single_neuron_spikes, cluster_name):
        """Compute a vector of ISIs for a single neuron given spike times.

        Args:
        spike_times (list of 1D arrays): Spike time dataset, with the first
        dimension corresponding to different neurons.
        neuron_idx (int): Index of the unit to compute ISIs for.

        Returns:
        isis (1D array): Duration of time between each spike from one neuron.
        """

        # Compute the ISIs for this set of spikes
        # Hint: the function np.diff computes discrete differences along an array
        isis = np.diff(single_neuron_spikes)

        fig, ax = plt.subplots()
        
        ax.hist(isis, bins=50, histtype="stepfilled")
        ax.axvline(isis.mean(), color="orange", label="Mean ISI")
        ax.set_xlabel("ISI duration (20kHz)")
        ax.set_ylabel("Number of spikes")
        #ax.set_title(f'Cluster:{cluster_name}')
        ax.legend()

        return fig, ax

    #spike trains========================
    # plot spike trains for all trials
    def plt_spike_train(self, cluster_name):
        fig, ax = plt.subplots()
        neuron_idx = self.get_neuron_idx_from_cluster_name(cluster_name)
        spikes_per_trial = self.spikes_per_trial_ar[neuron_idx]

        # plot spike trains
        ax.eventplot(spikes_per_trial, color=".2")        
        #plot prob change
        x_min = -100
        x_max= ax.get_xlim()[1]
        for group, frame in self.selected_trials_df.groupby('probability',sort=False):
            ax.hlines(frame.index[0], x_min, x_max, colors='r',linestyle='--',linewidths=(1,))
            ax.text(ax.get_xlim()[1]-14000, frame.index[0]+2, f"{group}%", fontsize=10)
        # clean up axis labels and ticks
        ticks=np.arange(0,(np.concatenate(spikes_per_trial).ravel()).max(),20000,dtype=int)
        labels=((ticks/20000).astype(int)).astype(str).tolist()
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Trial Length [s]')
        ax.set_ylabel('Trial')

        #ax.set_title(f"Spike Train of Trials for Cluster: {cluster_name}")
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        return fig, ax


    # plot spike trains and histogram to subplots
    def plt_spike_train_hist(self, cluster, selected_trials_df, event, window, bins, fig=None, ax=[None, None], title=None):
        """
        def:    plot the spike train around event (0) for all trials stacked on each other for event +/- delta
                and the histogram for the count of spikes over all trials
        params: cluster= integer::cluster (aka Neuron) to plot spikes for
                selected_trials= DataFrame::dataframe with all the trials to plot
                event= string::event in question (must be in trials_df as column name)
                window = integer::half window width in milli seconds
                bins = number of bins for histogram
                fig = pyplot subfigure, if None => will create one
                ax = dict of at least two pyplot subfigure axis, if None => will create one
                title = alternative subtitle
        return: plot
        """
        spikes = self.spikes_df[self.spikes_df.loc[:]['cluster'] == cluster]['spike_times']
        trials = selected_trials_df[event]
        delta = window*20

        # create plot and axis if none is passed
        if any(i==None for i in ax)or fig==None:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'hspace': 0})

        # loop that iterats trough all indeces in trial df
        y=0
        prop = selected_trials_df.iloc[0]['probability']
        prop_li = []
        # get x upper lim

        for row in trials.index:
            # define length of spike for row
            ypos = [y, y+1]
            y+=1
            # derive spike times in range delta around event time for trial
            ar = spikes[( ( spikes >= (trials[row] - delta) ) & ( spikes <= (trials[row] + delta) ) )].values
            ar = ar.astype('int64')
            ar = ar - trials[row]
            if ar.size > 0:
                #append to historam data frame
                if 'hist_sp' in locals():
                    hist_sp = np.append(hist_sp, ar)
                else:
                    hist_sp = ar
                # iterate trough all elements of np array
            for col in ar:
                ## plot spike train=========================
                ax[0].plot([col, col], ypos, 'k-', linewidth=0.8)

            # plot probability
            current_prop = selected_trials_df.loc[row]['probability']
            if current_prop != prop:
                prop = current_prop
                prop_li.append((prop,y))

        #x_lim_min = ax[0].get_xlim()[0]
        #x_lim_max = ax[0].get_xlim()[1] #trials[row] + delta#ax[0].get_xlim()[1]
        #_text = x_lim-2500
        #x_text = delta*2#-2500       
        #ax[0].text(x_text, 0+2, f"{selected_trials_df.iloc[0]['probability']}%", fontsize=10)
        for po, yp in prop_li:
            ax[0].hlines(yp, -delta, delta, colors='r',linestyle='--',linewidths=0.8)
            ax[0].text(delta+400, yp-4, f"{po*100}%", fontsize=10)#, colors='r')

        ## traw red line at event ==============
        ax[0].axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # spike train y lable
        ax[0].set_ylabel('Trial')

        # axis 0
        # set ticks
        step = trials.index.size/5
        start = 0
        stop = trials.index.size+step/2
        ax[0].set_yticks(np.arange(start, stop, step).astype(int))
        # set tick labels
        stop = trials.index.size
        label = trials.index.values[np.arange(start, stop, step).astype(int)]
        label = np.append(label, trials.index.values[-1])
        ax[0].set_yticklabels(label)
        # set y limits 1. plot
        ax[0].set_ylim([0, stop])

        #labels
        # specify y tick distance
        #ax[0].set_yticks(trials_df.index[trials_df['select'] == True][0::30])
        # trun x labels inside
        ax[0].tick_params(axis="x",direction="in")
        # turn of labels on shared x axis only ticks
        plt.setp(ax[0].get_xticklabels(), visible=False)
        # write event
        ax[0].set_title(event, color='red', fontsize=8)

        ## plot histogram===========================
        # draw histogram
        if 'hist_sp' in locals():
            ax[1].hist(hist_sp, bins=bins)
        # draw red line at event
        ax[1].axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # naming y axis
        ax[1].set_ylabel('Spike Count')
        # set x ticks to seconds
        if window > 1000:
            window = window/1000
        step = window/4
        start = -window
        stop = window+(step/2)
        y_index = np.arange(start, stop, step)
        ax[1].set_xticklabels(y_index)
        # set ticks top and bottom
        ax[1].tick_params(axis='x', bottom=True, top=True)
        # set x limits
        ax[1].set_xlim([-delta, delta])
        #ax.set_title('Spikes for Cluster 1')
        #if title != None:
        #    event = title
        #fig.suptitle(f"Spikes for Cluster: {cluster}")# at Event: {event}")
        # naming
        plt.xlabel('Window [s]')
        # if save = True -> save to path
        if 'hist_sp' not in locals():
            hist_sp = 0
        return fig, ax, hist_sp 

    def _test_plt_spike_train_hist(self, cluster, selected_trials, event, window, bins, fig=None, ax=[None, None], title=None):
        """
        def:    plot the spike train around event (0) for all trials stacked on each other for event +/- delta
                and the histogram for the count of spikes over all trials
        params: cluster= integer::cluster (aka Neuron) to plot spikes for
                selected_trials= DataFrame::dataframe with all the trials to plot
                event= string::event in question (must be in trials_df as column name)
                window = integer::half window width in milli seconds
                bins = number of bins for histogram
                fig = pyplot subfigure, if None => will create one
                ax = dict of at least two pyplot subfigure axis, if None => will create one
                title = alternative subtitle
        return: plot
        """
        cluster_df = self.spikes_df[self.spikes_df.loc[:]['cluster'] == cluster]['spike_times']
        trials = selected_trials[event]
        delta = window*20

        # create plot and axis if none is passed
        if any(i==None for i in ax)or fig==None:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'hspace': 0})
        else:
            ax1, ax2 = ax
        # loop that iterats trough all indeces in trial df
        y=0
        for row in trials.index:
            # define length of spike for row
            ypos = [y, y+1]
            y+=1
            # derive spike times in range delta around event time for trial
            ar = cluster_df[( ( cluster_df >= (trials[row] - delta) ) & ( cluster_df <= (trials[row] + delta) ) )].values
            ar = ar.astype('int64')
            ar = ar - trials[row]
            if ar.size > 0:
                #append to historam data frame
                if 'hist_sp' in locals():
                    hist_sp = np.append(hist_sp, ar)
                else:
                    hist_sp = ar
                # iterate trough all elements of np array
                for col in ar:
                    ## plot spike train=========================
                    ax1.plot([col, col], ypos, 'k-', linewidth=0.8)

        ## traw red line at event ==============
        ax1.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # spike train y lable
        ax1.set_ylabel('Trial')

        # axis 0
        # set ticks
        step = trials.index.size/5
        start = 0
        stop = trials.index.size+step/2
        ax1.set_yticks(np.arange(start, stop, step).astype(int))
        # set tick labels
        stop = trials.index.size
        label = trials.index.values[np.arange(start, stop, step).astype(int)]
        label = np.append(label, trials.index.values[-1])
        ax1.set_yticklabels(label)
        # set y limits 1. plot
        ax1.set_ylim([0, stop])

        #labels
        # specify y tick distance
        #ax[0].set_yticks(trials_df.index[trials_df['select'] == True][0::30])
        # trun x labels inside
        ax1.tick_params(axis="x",direction="in")
        # turn of labels on shared x axis only ticks
        plt.setp(ax1.get_xticklabels(), visible=False)
        # write event
        ax.set_title(event, color='red', fontsize=8)

        ## plot histogram===========================
        # draw histogram
        ax2.hist(hist_sp, bins=bins)
        # draw red line at event
        ax2.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # naming y axis
        ax2.set_ylabel('Spike Count')
        # set x ticks to seconds
        if window > 1000:
            window = window/1000
        step = window/4
        start = -window
        stop = window+(step/2)
        y_index = np.arange(start, stop, step)
        ax2.set_xticklabels(y_index)
        # set ticks top and bottom
        ax2.tick_params(axis='x', bottom=True, top=True)
        # set x limits
        ax2.set_xlim([-delta, delta])
        #ax.set_title('Spikes for Cluster 1')
        #if title != None:
        #    event = title
        #fig.suptitle(f"Spikes for Cluster: {cluster} at Event: {event}")
        # naming
        plt.xlabel('Sampling Points [ms]')
        # if save = True -> save to path
        return fig, ax
        

    def plt_spike_train_hist_all_events(self, cluster, selected_trials_df, event, window, bins, fig=None, ax=[None, None], title=None):
        spikes = self.spikes_df[self.spikes_df.loc[:]['cluster'] == cluster]['spike_times']
        trials = selected_trials_df[event]
        delta = window*20

        # create plot and axis if none is passed
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'hspace': 0})

        # loop that iterats trough all indeces in trial df
        y=0
        prop = selected_trials_df.iloc[0]['probability']
        prop_li = []
        # get x upper lim

        for row in trials.index:
            # define length of spike for row
            ypos = [y, y+1]
            y+=1
            # derive spike times in range delta around event time for trial
            ar = spikes[( ( spikes >= (trials[row] - delta) ) & ( spikes <= (trials[row] + delta) ) )].values
            ar = ar.astype('int64')
            ar = ar - trials[row]
            if ar.size > 0:
                #append to historam data frame
                if 'hist_sp' in locals():
                    hist_sp = np.append(hist_sp, ar)
                else:
                    hist_sp = ar
                # iterate trough all elements of np array
            for col in ar:
                ## plot spike train=========================
                ax[0].plot([col, col], ypos, 'k-', linewidth=0.8)
            
            # plott all other events
            all_events  = selected_trials_df.loc[row,['start','cue','sound','openloop', 'reward', 'iti']] - trials[row]
            for ev in all_events:
                # only plot not event events
                if ev != 0:
                    ax[0].plot([ev, ev], ypos, c="red",linewidth=0.5)

            # plot probability
            current_prop = selected_trials_df.loc[row]['probability']
            if current_prop != prop:
                prop = current_prop
                prop_li.append((prop,y))

        
        # write all other events
        for ev_time,ev_name in zip(all_events,['start','cue','sound','openloop', 'reward', 'iti']):
                # only plot not event events
                ax[0].text(ev_time-5, y+10, ev_name, c='red', fontsize=5, rotation='vertical',)
        
        # plot prop
        for po, yp in prop_li:
            ax[0].hlines(yp, -delta, delta, colors='r',linestyle='--',linewidths=0.8)
            ax[0].text(delta+400, yp-4, f"{po*100}%", fontsize=10)#, colors='r')
        #print(prop_li)

        ## traw red line at event ==============
        ax[0].axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # spike train y lable
        ax[0].set_ylabel('Trial')

        # axis 0

        # axis 0
        # set ticks
        step = trials.index.size/5
        start = 0
        stop = trials.index.size+step/2
        ax[0].set_yticks(np.arange(start, stop, step).astype(int))
        # set tick labels
        stop = trials.index.size
        label = trials.index.values[np.arange(start, stop, step).astype(int)]
        label = np.append(label, trials.index.values[-1])
        ax[0].set_yticklabels(label)
        # set y limits 1. plot
        ax[0].set_ylim([0, stop])

        #labels
        # specify y tick distance
        #ax[0].set_yticks(trials_df.index[trials_df['select'] == True][0::30])
        # trun x labels inside
        ax[0].tick_params(axis="x",direction="in")
        # turn of labels on shared x axis only ticks
        plt.setp(ax[0].get_xticklabels(), visible=False)
        # write event
        #ax[0].set_title(event, color='red', fontsize=8,rotation='vertical')

        ## plot histogram===========================
    
        # draw histogram
        if 'hist_sp' in locals():
            ax[1].hist(hist_sp, bins=bins)
        # draw red line at event
        ax[1].axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # naming y axis
        ax[1].set_ylabel('Spike Count')
        # set x ticks to seconds
        if window > 1000:
            window = window/1000
        step = window/4
        start = -window
        stop = window+(step/2)
        y_index = np.arange(start, stop, step)
        ax[1].set_xticklabels(y_index)
        # set ticks top and bottom
        ax[1].tick_params(axis='x', bottom=True, top=True)
        # set x limits
        ax[1].set_xlim([-delta, delta])
        #ax.set_title('Spikes for Cluster 1')
        #if title != None:
        #    event = title
        #fig.suptitle(f"Spikes for Cluster: {cluster}",y=1.02)# at Event: {event}",)
        # naming
        plt.xlabel('Window [s]')
        # if save = True -> save to path

        return fig, ax

    # plot spike train, histogram for bin and histogram for trials
    def plt_spike_train_hist_bar(self, cluster, selected_trials, event, window, bins, fig=None, ax=[None, None, None], title=None):
        # create necessary variables
        cluster_df = self.spikes_df[self.spikes_df.loc[:]['cluster'] == cluster]['spike_times']
        trials = selected_trials[event]
        delta = window*20
        # create fig, gird and axis
        if any(i==None for i in ax)or fig==None:
            #create figure with shape
            fig = plt.figure(figsize=(6,5))
            # create gridspecs
            gs = fig.add_gridspec(2, 3,  hspace=0, wspace=0)
            # create axis for hist spike train
            ax1 = fig.add_subplot(gs[0, :2])
            ax2 = fig.add_subplot(gs[1, :2])
            ax2.get_shared_x_axes().join(ax1, ax2)
            # create axis for trial hist
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.get_shared_y_axes().join(ax1, ax3)
        else:
            ax1, ax2, ax3 = ax
        # loop that iterats trough all indeces in trial df
        y = 0
        # loop for hist trial plot
        hist_tr = pd.DataFrame(columns=['spike count'])
        hist_tr.index.name = 'trial'

        ##spike train plot ========================
        # main loop over each trial
        for row in trials.index:
            # define length of spike for row
            ypos = [y, y+1]
            y+=1
            # derive spike times in range delta around event time for trial
            ar = cluster_df[( ( cluster_df >= (trials[row] - delta) ) & ( cluster_df <= (trials[row] + delta) ) )].values
            ar = ar.astype('int64')
            ar = ar - trials[row]
            # create hist trial dataframe
            series = pd.Series([ar.size], index=['spike count'])
            series.name = row
            hist_tr = hist_tr.append(series)
            # add to histogram array
            if ar.size > 0:
                #append to historam data frame
                if 'hist_sp' in locals():
                    hist_sp = np.append(hist_sp, ar)
                else:
                    hist_sp = ar
                # iterate trough all elements of np array
                for col in ar:
                    ## plot spike train=========================
                    ax1.plot([col, col], ypos, 'k-', linewidth=0.8)
        ## traw red line at event
        ax1.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # spike train y lable
        ax1.set_ylabel('Trial')
        ## set y axis 1. plot
        # set ticks
        step = trials.index.size/5
        start = 0
        stop = trials.index.size+step/2
        ax1.set_yticks(np.arange(start, stop, step).astype(int))
        # set tick labels
        stop = trials.index.size
        label = trials.index.values[np.arange(start, stop, step).astype(int)]
        label = np.append(label, trials.index.values[-1])
        ax1.set_yticklabels(label)
        # set y limits 1. plot
        ax1.set_ylim([0, stop])
        ##labels
        # trun x labels inside
        ax1.tick_params(axis="x",direction="in")
        # turn of labels on shared x axis only ticks
        plt.setp(ax1.get_xticklabels(), visible=False)
        # write event
        ax1.set_title(event, color='red', fontsize=8)

        ## plot histogram spikes ===========================
        # draw histogram
        ax2.hist(hist_sp, bins=bins, color="tab:blue")
        # draw red line at event
        ax2.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # naming y axis
        ax2.set_ylabel('Spike Count')
        # set x ticks
        step = delta/4
        start = -delta
        stop = delta+(step/2)
        x_ticks = np.arange(start, stop, step)
        ax2.set_xticks(x_ticks)
        # set x ticks labels to seconds
        if window > 1000:
            window = window/1000
        step = window/4
        start = -window
        stop = window+(step/2)
        x_labels = np.arange(start, stop, step)
        ax2.set_xticklabels(x_labels)
        # set ticks top and bottom
        ax2.tick_params(axis='x', bottom=True, top=True)
        # set x limits
        ax2.set_xlim([-delta, delta])

        ## plot histogram trials =================================
        #pos = hist_tr.index.values
        pos = np.arange(0, hist_tr.size).astype(float)
        #values
        values = hist_tr.values.reshape([hist_tr.values.size]).astype(float)
        # invert axis
        ax3.invert_xaxis()
        # remove ticks
        ax3.set_yticks([])

        ## plot histogram
        ax3 = plt.barh(pos, values, height=1.0, color='lightgray')

        # name main title
        #ax.set_title('Spikes for Cluster 1')
        if title != None:
            event = title
        fig.suptitle(f"Spikes for Cluster: {cluster} at Event: {event}")
        # naming
        plt.xlabel('Position [ms]')

        return fig, (ax1, ax2, ax3)


    # firing frequency binned ==========


# Plot and Save all figures ===================================================================================================

    def save_fig(self, name, fig):
        folder = self.folder+"/figures/all_figures"
        try:
            fig.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')
        except:
            fig[0].savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')
        


    def generate_plots(self,window,bins):
        """
        #hist and fit
        print("plot hist and fit")
        fig,ax=self.plt_trial_hist_and_fit(self.selected_trials_df.loc[:,'length'], bins)
        self.save_fig('hist_fit',fig)
        
        # trial length
        print("trial lengt")
        fig,ax=plt.subplots()
        ax.plot(self.selected_trials_df.loc[:,'length'])
        ax.set_ylabel('length [ms]')
        ax.set_xlabel('trial')
        self.save_fig('trial_length',fig)
        
        # cluster histogram
        print("cluster histogram")
        fig,ax=self.plt_all_cluster_spikes_hist()
        self.save_fig('cluster_hist',fig)
        
        # plott all isi for good clustes only focus between selected trials
        print("inter spike interval -> all")
        start = self.selected_trials_df.loc[0,'start']
        end = self.selected_trials_df.iloc[-1]['end']
        
        for cluster, row in self.clusters_df.loc[self.clusters_df['group']=='good'].iterrows():
            a = row['spikes']
            fig,ax=self.plot_single_neuron_isis(a[np.logical_and(a>=start, a<=end)],cluster)
            self.save_fig('isi_'+str(cluster),fig)
        
        # spike trains
        print("spike trains -> all")
        for cluster in self.clusters_df.loc[self.clusters_df['group']=='good'].index:
            fig,ax=self.plt_spike_train(cluster)
            self.save_fig('spk_train_'+str(cluster),fig)
        """    
        # spike train + hist all trials
        print("cue aligned - spike trains + histogram -> all")
        for cluster in self.clusters_df.index:
            fig,ax = self.plt_spike_train_hist_all_events(cluster, self.selected_trials_df, 'cue', window, bins)
            self.save_fig('spk_train_hist_all-events_'+str(cluster),fig)
        
        print("reward algined - spike trains + histogram -> all")
        for cluster in self.clusters_df.index:
            fig,ax = self.plt_spike_train_hist_all_events(cluster, self.selected_trials_df, 'reward', window, bins)
            self.save_fig('spk_train_hist_all-events_reward-centered_'+str(cluster),fig)
        
        # plott reward at specific trials
        # get gambl side
        if self.gamble_side == 'right':
            save='left'
            gamble='right'
        else:
            save='right'
            gamble='left'
        
        #spike train + hist reward specific events
        trials_subselctor_list = []
        filenames_subselect_list = []
        # rewarded ===
        # reward right
        print("reward + gamble - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[(self.selected_trials_df[gamble])&(self.selected_trials_df['reward_given'])]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000,bins)
            self.save_fig('spk_train_hist_gamble_reward_'+str(cluster),fig)
        
        # reward left
        print("reward + safe - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[(self.selected_trials_df[save])&(self.selected_trials_df['reward_given'])]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_save_reward_'+str(cluster),fig)
        
        # not rewarded ===
        # gamble + norw
        print("no-reward + gamble - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[(self.selected_trials_df[gamble])&(np.invert(self.selected_trials_df['reward_given']))]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_gamble_no-reward_'+str(cluster),fig)
        
        # safe + norw
        print("no-reward + safe - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[(self.selected_trials_df[save])&(np.invert(self.selected_trials_df['reward_given']))]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_save_no-reward_'+str(cluster),fig)
            
        
        # all reward
        print("reward - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[self.selected_trials_df['reward_given']]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_reward_'+str(cluster),fig)
            
        # all right
        print("gamble - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[self.selected_trials_df[gamble]]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_gamble_'+str(cluster),fig)

        # all left
        print("safe - spike trian + histogram -> all")
        selected_trials = self.selected_trials_df[self.selected_trials_df[save]]
        for cluster in self.clusters_df.index:
            fig,ax,_ = self.plt_spike_train_hist(cluster, selected_trials, 'reward', 2000, bins)
            self.save_fig('spk_train_hist_save_'+str(cluster),fig)