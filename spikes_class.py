import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os

# for pylatex
from pylatex import Document, Section, Subsection, Command, Package, NewPage, LongTabu, Tabular
from pylatex.utils import italic, NoEscape

class SpikeAnalysis():
    def __init__(self, session, main_folder, gamble_side):
        self.folder = main_folder
        self.spikes_df, self.clusters_df, self.trials_df = self.load_files()
        self.session = session
        self.gamble_side = gamble_side

    # load files from kilosort & behavior files
    def load_files(self):
        """

        """
        if platform.system() == 'Linux':
            # load fies in liux
            spike_times = np.load(self.folder+r"/electrophysiology/spike_times.npy")
            spike_cluster = np.load(self.folder+r"/electrophysiology/spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"/electrophysiology/cluster_info.tsv", sep='\t')
            excel_df = pd.read_excel(self.folder+"/output_file.xlsx", 'Daten', header=[0, 1] )

        elif platform.system() == 'Windows':
            # load files in Windows
            spike_times = np.load(self.folder+r"\electrophysiology\spike_times.npy")
            spike_cluster = np.load(self.folder+r"\electrophysiology\spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"\electrophysiology\cluster_info.tsv", sep='\t')
            excel_df = pd.read_excel(self.folder+r"\output_file.xlsx", 'Daten', header=[0, 1] )

        elif platform.system() == 'Darwin':
            # load fies in liux
            spike_times = np.load(self.folder+r"/electrophysiology/spike_times.npy")
            spike_cluster = np.load(self.folder+r"/electrophysiology/spike_clusters.npy")
            clusters_df = pd.read_csv(self.folder+r"/electrophysiology/cluster_info.tsv", sep='\t')
            excel_df = pd.read_excel(self.folder+"/output_file.xlsx", 'Daten', header=[0, 1] )

        # create spike Data Frame with clusters and spike times
        spikes_df = pd.DataFrame( { 'cluster':spike_cluster, 'spike_times': spike_times[:,0] } )
        spikes_df.index.name = 'global index'
        # set indexes for each clusters
        spikes_df = spikes_df.set_index((spikes_df.groupby('cluster').cumcount()).rename('cluster index'), append=True)
        # clean up cluster data frame
        clusters_df = clusters_df.rename(columns={'id':'cluster id'})
        clusters_df = clusters_df.set_index('cluster id')
        # create 'spikes' colum with spiketimes
        spk = pd.DataFrame( {'spikes':np.zeros(clusters_df.shape[0], dtype=object)}, index=clusters_df.index )
        for group, frame in spikes_df.groupby('cluster'):
            spk['spikes'][group] = frame['spike_times'].values
        #merge spike column with clusters_df
        clusters_df = pd.merge(clusters_df, spk, how='right', left_index=True, right_index=True)

        # clean up trials data Frame
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
        trials_df['select']=np.full((trials_df.shape[0] ,1), True ,dtype='bool')
        return (spikes_df, clusters_df, trials_df)



# Helper Functions EDA =================================================================================================
    # find spikes between
    def get_spikes_for_trial(self, array, start, stop): #old
        '''
        params: array = numpy array (N,1) with values to check against
                start, stop = value to find all values in array between
        return: valus in array between start and stop
        '''
        ar = array[np.logical_and(array >= start, array <= stop)].values
        if ar.size > 0:
            ar = ar[:] - ar[0]
        return a
    def new_get_spikes_for_trial(array, start, stop): #new
        '''
        params: array = numpy array (N,1) with values to check against
                start, stop = value to find all values in array between
        return: valus in array between start and stop
        '''
        ar = array[np.logical_and(array >= start, array <= stop)]
        if ar.size > 0:
            ar = ar[:] - ar[0]
        return ar

    # get all spikes for specified clusters
    def get_spikes_for_cluster(trials_df, cluster):
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
            df1 = pd.DataFrame({row:get_spikes_for_trial(cluster, start, stop)}, dtype="Int64")
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
        bins = self.trials_df['end'].values
        bins = np.insert(bins, 0, 0)
        # labels
        labels = self.trials_df.index.values
        # add trial index
        df['trial'] = pd.cut(df['bin end time'], bins, labels=labels, right=True, include_lowest=True)
        df.set_index('trial', append=True, inplace=True)
        df = df.swaplevel(0, 1)
        return df

    #  Compute a vector of ISIs for a single neuron given spike times.
    def compute_single_neuron_isis(spike_times, neuron_idx):
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

# Plotting ==========================================================================================================

    # plot histogram of trial times & normal fitted cuve
    def plt_trial_hist_and_fit(self, df):
        fig, ax = plt.subplots()
        # plot histogramm
        num_bins = 50
        n, bins, patches = ax.hist(df, num_bins, density=1)
        # add a 'best fit' line
        mean = df.mean()
        std = df.std()
        y = st.norm.pdf(bins, df.mean(), df.std())
        ax.plot(bins, y, '-')
        ax.axvline(x=mean, color='y')
        ax.set_xlabel('Trial Length')
        ax.set_ylabel('Probability density')
        ax.set_title(f"Histogram of Trial Length $\mu=${round(mean, 2)}, $\sigma=${round(std, 4)}")
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        return fig, ax

    # plot spike times
    def plt_trial_length(self):
        fig, ax = plt.subplots()
        ax.plot(self.trials_df.loc[self.trials_df.loc[:,'select'],'length'])

    # plot inter spike interval

    #spike trains========================
    # plot spike trains for all trials
    def plt_spike_train(self, cluster, trials_df): #old
        """
        def: plots the spike trins fro each trial stacked on top of each other
        params: cluster =
        """
        # initialize plot
        fig, ax = plt.subplots()

        # get spikes for each trial for this cluster
        for row in trials_df.index[trials_df['select'] == True]:
            ypos = [row, row+0.8]
            start = trials_df.loc[row, 'start']
            stop = trials_df.loc[row, 'end']
            for col in self.get_spikes_for_trial(cluster, start, stop):
                ax.plot([col, col], ypos)

        ax.set_title('Spikes for Cluster 1')
        ax.set_xlabel('Sampling Points [20kHz]')
        ax.set_ylabel('Trial')
        plt.yticks(trials_df.index[trials_df['select'] == True][0::10])

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        return fig, ax
    def new_plt_spike_train(self, cluster, trials_df): #new
        """
        def: plots the spike trins fro each trial stacked on top of each other
        params: cluster =
        """    
        # initialize plot
        fig, ax = plt.subplots()
        # initialize list with spikes per trial
        spikes_trials = []
        # get spikes for each trial
        for row in self.trials_df.index[self.trials_df['select'] == True]:
            start = self.trials_df.loc[row, 'start']
            stop = self.trials_df.loc[row, 'end']
            spk = self.new_get_spikes_for_trial(self.clusters_df.loc[cluster]['spikes'], start, stop)
            #if len(spk)>0:
            spikes_trials.append(spk)
        # plot spikes
        ax.eventplot(spikes_trials, color=".2")
        # set title and axis labels
        ax.set_title('Spikes for Cluster 1')
        ax.set_xlabel('Sampling Points [20kHz]')
        ax.set_ylabel('Trial')
        index = self.trials_df[self.trials_df['select'] == True].index[0::10]
        ax.set_yticks(index - index[0])
        ax.set_yticklabels(index)
        return ax, fig

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show
        
    
    # plot spike trains and histogram to subplots
    def plt_spike_train_hist(self, cluster, selected_trials, event, window, fig=None, ax=[None, None], title=None):
        """
        def:    plot the spike train around event (0) for all trials stacked on each other for event +/- delta
                and the histogram for the count of spikes over all trials
        params: cluster= integer::cluster (aka Neuron) to plot spikes for
                selected_trials= DataFrame::dataframe with all the trials to plot
                event= string::event in question (must be in trials_df as column name)
                window = integer::half window width in milli seconds
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
                    ax[0].plot([col, col], ypos, 'k-', linewidth=0.8)

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
        num_bins = 60
        # draw histogram
        ax[1].hist(hist_sp, bins=num_bins)
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
        if title != None:
            event = title
        fig.suptitle(f"Spikes for Cluster: {cluster} at Event: {event}")
        # naming
        plt.xlabel('Sampling Points [ms]')
        # if save = True -> save to path
        return fig, ax

    def _test_plt_spike_train_hist(self, cluster, selected_trials, event, window, fig=None, ax=[None, None], title=None):
        """
        def:    plot the spike train around event (0) for all trials stacked on each other for event +/- delta
                and the histogram for the count of spikes over all trials
        params: cluster= integer::cluster (aka Neuron) to plot spikes for
                selected_trials= DataFrame::dataframe with all the trials to plot
                event= string::event in question (must be in trials_df as column name)
                window = integer::half window width in milli seconds
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
        num_bins = 60
        # draw histogram
        ax2.hist(hist_sp, bins=num_bins)
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
        if title != None:
            event = title
        fig.suptitle(f"Spikes for Cluster: {cluster} at Event: {event}")
        # naming
        plt.xlabel('Sampling Points [ms]')
        # if save = True -> save to path
        return fig, ax

    # plot spike train, histogram for bin and histogram for trials
    def plt_spike_train_hist_bar(self, cluster, selected_trials, event, window, fig=None, ax=[None, None, None], title=None):
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
        num_bins = 60
        # draw histogram
        ax2.hist(hist_sp, bins=num_bins, color="tab:blue")
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



# Save all & Create Report ===================================================================================================
    # save images of spike train and histogram plot for all good clusters for all reward event
    def save_plt_spike_train_hist_reward(self, window, update=False):
        """
        def:    saves the spike train for all trials stacked on each other for event +/- window in seconds
                and the histogram for the count of spikes over all trials
                for all clusters
        params: window = delta window in ms
        return:
        """
        # batch plot for single cluster all reward configurations
        # gamble side = right -> reward = 5, no reward = 6
        # reward
        sel_tr_rw_all = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==5) | (self.trials_df['event']==7) ) ][:]
        sel_tr_rw_gambl = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==5) ) ][:]
        sel_tr_rw_save = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==7) ) ][:]
        # no reward
        sel_tr_norw_all = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==8) | (self.trials_df['event']==6) ) ][:]
        sel_tr_norw_gambl = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==6) ) ][:]
        sel_tr_norw_save = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==8) ) ][:]
        # title
        tlt_rw_all = ("reward (both sides)")
        tlt_rw_gambl = ("reward gambl side")
        tlt_rw_save = ("reward save side")
        tlt_norw_all = ("no-reward (both sides)")
        tlt_norw_gambl = ("no-reward gambl side")
        tlt_norw_save = ("no-reward save side")
        # file name
        fln_rw_all = ("reward")
        fln_rw_gambl = ("reward-gambl")
        fln_rw_save = ("reward-save")
        fln_norw_all = ("no-reward")
        fln_norw_gambl = ("no-reward-gambl")
        fln_norw_save = ("no-reward-save")
        # touples (selected trials dataframe, name, file name ending)
        rw_all = (sel_tr_rw_all, tlt_rw_all, fln_rw_all)
        rw_gambl = (sel_tr_rw_gambl, tlt_rw_gambl, fln_rw_gambl)
        rw_save = (sel_tr_rw_save, tlt_rw_save, fln_rw_save)
        norw_all = (sel_tr_norw_all, tlt_norw_all, fln_norw_all)
        norw_gambl = (sel_tr_norw_gambl, tlt_norw_gambl, fln_norw_gambl)
        norw_save = (sel_tr_norw_save, tlt_norw_save, fln_norw_save)
        # pack all to list
        plots = [rw_all, rw_gambl, rw_save, norw_all, norw_gambl, norw_save]

        #=========================================================================================================

        # load path to pictures for os
        if platform.system() == 'Linux':
            path = (self.folder + r"/figures/spikes/spike-train-hist-event" )
        elif platform.system() == 'Windows':
            path = (self.folder + r"\figures\spikes\spike-train-hist-event" )
        # check if folder exists if not create
        if not os.path.isdir(path):
            os.makedirs(path)

        # iterate over all clusters
        for cluster in self.clusters_df.loc[self.clusters_df['group']=='good'].index:
            # iterate over all different reward events
            for plot in plots:
                # unpack touples
                selected_trials, title, file_name = plot
                # create filename
                if platform.system() == 'Linux':
                    name = (r"/cluster-" + str(cluster) + "-" + file_name + ".png")
                elif platform.system() == 'Windows':
                    name = (r"\cluster-" + str(cluster) + "-" + file_name + ".png")
                file = (path + name)
                # create subplot
                fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'hspace': 0})
                # plot figure
                fig, axs = self.plt_spike_train_hist(cluster, selected_trials, 'reward', window, fig, axs, title)
                # save figure
                plt.savefig(file, dpi=300)
                plt.close(fig)

                #print infos
                print(f"\tplot {title} finished")
            print(f"cluster {cluster} finished")
        print(f"\nall plots finished")

    # save images of spike train and histogram and bar plot for all good clusters for all reward events
    def save_plt_spike_train_hist_bar_reward(self, window, update=False):
        """
        def:    saves the spike train for all trials stacked on each other for event +/- window in seconds
                and the histogram for the count of spikes over all trials
                for all clusters
        params: window = delta window in ms
        return:
        """
        # batch plot for single cluster all reward configurations
        # gamble side = right -> reward = 5, no reward = 6
        # reward
        sel_tr_rw_all = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==5) | (self.trials_df['event']==7) ) ][:]
        sel_tr_rw_gambl = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==5) ) ][:]
        sel_tr_rw_save = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==7) ) ][:]
        # no reward
        sel_tr_norw_all = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==8) | (self.trials_df['event']==6) ) ][:]
        sel_tr_norw_gambl = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==6) ) ][:]
        sel_tr_norw_save = self.trials_df.loc[ (self.trials_df['select'] == True) & ( (self.trials_df['event']==8) ) ][:]
        # title
        tlt_rw_all = ("reward (both sides)")
        tlt_rw_gambl = ("reward gambl side")
        tlt_rw_save = ("reward save side")
        tlt_norw_all = ("no-reward (both sides)")
        tlt_norw_gambl = ("no-reward gambl side")
        tlt_norw_save = ("no-reward save side")
        # file name
        fln_rw_all = ("reward")
        fln_rw_gambl = ("reward-gambl")
        fln_rw_save = ("reward-save")
        fln_norw_all = ("no-reward")
        fln_norw_gambl = ("no-reward-gambl")
        fln_norw_save = ("no-reward-save")
        # touples (selected trials dataframe, name, file name ending)
        rw_all = (sel_tr_rw_all, tlt_rw_all, fln_rw_all)
        rw_gambl = (sel_tr_rw_gambl, tlt_rw_gambl, fln_rw_gambl)
        rw_save = (sel_tr_rw_save, tlt_rw_save, fln_rw_save)
        norw_all = (sel_tr_norw_all, tlt_norw_all, fln_norw_all)
        norw_gambl = (sel_tr_norw_gambl, tlt_norw_gambl, fln_norw_gambl)
        norw_save = (sel_tr_norw_save, tlt_norw_save, fln_norw_save)
        # pack all to list
        plots = [rw_all, rw_gambl, rw_save, norw_all, norw_gambl, norw_save]

        #=========================================================================================================

        # load path to pictures for os
        if platform.system() == 'Linux':
            path = (self.folder + r"/figures/spikes/spike-train-hist-bin-event" )
        elif platform.system() == 'Windows':
            path = (self.folder + r"\figures\spikes\spike-train-hist-bin-event" )
        # check if folder exists if not create
        if not os.path.isdir(path):
            os.makedirs(path)

        # iterate over all clusters
        for cluster in self.clusters_df.loc[self.clusters_df['group']=='good'].index:
            # iterate over all different reward events
            for plot in plots:
                # unpack touples
                selected_trials, title, file_name = plot
                # create filename
                if platform.system() == 'Linux':
                    name = (r"/cluster-" + str(cluster) + "-" + file_name + ".png")
                elif platform.system() == 'Windows':
                    name = (r"\cluster-" + str(cluster) + "-" + file_name + ".png")
                file = (path + name)
                # create subplot
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
                # pack axes
                axs = (ax1, ax2, ax3)
                # plot figure
                fig, axs = self.plt_spike_train_hist_bar(cluster, selected_trials, 'reward', window, fig, axs, title)
                # save figure
                plt.savefig(file, dpi=300)
                plt.close(fig)

                #print infos
                print(f"\tplot {title} finished")
            print(f"cluster {cluster} finished")
        print(f"\nall plots finished")

    # create latex report with all images
    def image_box(self, image, cluster, width=0.5, last=False):
        arg = (r"\parbox[c]{1em}{\includegraphics[width="+ str(width) + r"\textwidth]{" + r"/media/max/Max01/backu/figures" + r"/spikes/spike-train-hist-event/cluster-"+
        str(cluster) + "-" + image + r".png}}" )
        if last:
            arg += r"\\"
        else:
            arg += "&"
        return arg
    def create_report(self, included_sections):
        # Basic document
        # Document with `\maketitle` command activated
        doc = Document(default_filepath=(self.folder + r"/figures"))
        doc.documentclass = Command(
            'documentclass',
            options=['10pt', 'a4'],
            arguments=['article'],
        )
        doc.packages.append(NoEscape(r'\setcounter{tocdepth}{4}'))
        doc.packages.append(NoEscape(r'\setcounter{secnumdepth}{1}'))
        # usepackages
        doc.packages.append(Package('helvet'))
        doc.packages.append(Package('graphicx'))
        doc.packages.append(Package('geometry'))
        doc.packages.append(Package('float'))
        doc.packages.append(Package('amsmath'))
        doc.packages.append(Package('multicol'))
        doc.packages.append(Package('ragged2e'))
        doc.packages.append(Package('breakurl'))
        doc.packages.append(Package('booktabs, multirow'))
        doc.packages.append(Package('epstopdf'))
        doc.packages.append(NoEscape(r'\usepackage[nolist, nohyperlinks]{acronym}'))
        doc.packages.append(Package('hyperref'))


        # add commands
        doc.preamble.append(NoEscape(r"\renewcommand{\familydefault}{\sfdefault}"))
        doc.preamble.append(NoEscape(r"\newcommand\Tstrut{\rule{0pt}{3ex}}  % = `top' strut"))
        doc.preamble.append(NoEscape(r"\newcommand\Bstrut{\rule[1ex]{0pt}{0pt}}   % = `bottom' strut"))


        # make title
        title = "Report for Session: " + self.session
        doc.preamble.append(Command('title', title))
        #doc.preamble.append(Command('author', 'Anonymous author'))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))
        doc.append(NoEscape(r'\tableofcontents'))
        doc.append(NewPage())
        doc.append(NoEscape(r'\newgeometry{vmargin={12mm}, hmargin={10mm,10mm}}'))
        doc.append(NoEscape(r'\bigskip'))

        # Add stuff to the document
        if 'spike_trains_and_histogram' in included_sections:
            with doc.create(Section('Spike Trains and Histogram for Reward Events')):
                    # create necessary variables
                    for cluster in self.clusters_df.loc[self.clusters_df['group']=='good'].index[0:3]:
                        # create subsection title
                        subsection = "Cluster " + str(cluster)
                        with doc.create(Subsection(subsection, label=False)):
                            # create summary table
                            with doc.create(LongTabu("X | X")) as summary_table:
                                self.image_box("reward", cluster, 0.4)
                                with doc.create(Tabular("r r")) as small_table:
                                    small_table.add_row(["Summary",""])
                                    small_table.add_hline()
                                    small_table.add_row(["Gamble side:", self.gamble_side])

                            # create details table
                            with doc.create(LongTabu("X | X")) as details_table:
                                details_table.add_row(["Gambl Side", "Save Side"])
                                details_table.end_table_header()
                                details_table.add_row(["Reward", ""])
                                doc.append(NoEscape( self.image_box("reward-gambl", cluster) ))
                                doc.append(NoEscape( self.image_box("reward-save", cluster, last=True) ))
                                details_table.add_hline()
                                details_table.add_row(["No-reward", ""])
                                doc.append(NoEscape( self.image_box("no-reward-gambl", cluster) ))
                                doc.append(NoEscape( self.image_box("no-reward-save", cluster, last=True) ))
                        doc.append(NewPage())

        # create file_name
        filename = (self.session + "-Report")
        # create pdf
        doc.generate_pdf(filename, clean=True, clean_tex=False)#, compiler='latexmk -f -xelatex -interaction=nonstopmode')

    # create interactive webpage
