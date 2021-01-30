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

# for pylatex
from pylatex import Document, Section, Subsection, Command, Package, NewPage, LongTabu, Tabular
from pylatex.utils import italic, NoEscape

# numba helper functions
from numba import njit

@njit()
def create_random_start(trial_nr,iter_nr, trials_ar, delta):
    """get random event within trial

    Args:
        trial_nr (int): number of trials
        iter_nr (int): number of iterations
        trials_df (numpy ar): dataframe from spikes class with all trials
        delta (float): window = 2*delta

    Returns:
        random_li(numpy ar): array with random start points, (i=trial_nr, j=iter_nr)
    """
    #initialize complete dataframe
    random_li = np.zeros(shape=(trial_nr,iter_nr))
    #iterate over trials
    for i in range(trial_nr):
        random_li[i,:]=(np.random.randint((trials_ar[i,0]),(trials_ar[i,1]), size=(iter_nr)) )
    return random_li

# get spikes for event aligned windows =============================

def get_spikes_in_window(data_ar, i, delta):
    """get all spikes that fall into window(+-delta) around event i 

    Args:
        data_ar (np ar): spike times
        i (float): event time
        delta (float): 1/2 window width in sampling points

    Returns:
        np ar: array with spike times that are in window
    """
    return ((data_ar[( (data_ar>=(i-delta)) & (data_ar<=(i+delta)) )])-(i))

# fixed event
def get_spikes_in_window_all_trials_singlevent(spikes_ar, event_ar, delta):
    """get all spikes that fall in window of specific event for all trials

    Args:
        spikes_ar_all (np ar): array of arrays of spike times for each trial
        event_ar (np ar): array of event times for each trial
        delta (float): 1/2 window width in sampling points

    Returns:
        list: list of arrays with all spike times that fall in window for each trial
    """
    spikes_li_all = list()
    for trial_event in event_ar:
        spikes_li_all.append(get_spikes_in_window(spikes_ar, trial_event, delta))
    return spikes_li_all

# random events

def get_spikes_in_window_per_trial_all_randrang(data_ar, range_ar, delta):
    """get all spikes that fall in all random generated windows for specific trial

    Args:
        data_ar (numpy ar): spikes_per_trial_ar[cluster,trial,:] from spikes class
        range_ar (numpy ar): random_li output from create_random_start function

    Returns:
        list: list of arrays with spikes for all iterations for specific trial
    """
    results_li = list()
    # iterate ove all 1000 random events
    for i in range_ar:
        results_li.append(get_spikes_in_window(data_ar, i, delta))
    return results_li

def get_spikes_in_window_all_trial_all_randrang(spikes_ar, random_ar, delta):
    """get spikes for all trials and iterations

    Args:
        spikes_per_trial_df (np ar): spikes class matrix from
        random_ar (np ar): random events

    Returns:
        li: list of lists with all spikes for all trials and all iterations
    """ 
    spiketimes_li = list()
    #binnes_li = list()
    for i in range(random_ar.shape[0]):
        spiketimes_li.append(get_spikes_in_window_per_trial_all_randrang(spikes_ar, random_ar[i], delta))
    return spiketimes_li


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

def bin_trial_spike_times_single_cluster(input_ar,nr_bins):
    """binn randm windows from single clusters, all trials all iterations over complete trial

    Args:
        input_ar (np ar): spikes per random event for single clusters, all trials, all iterations
        nr_bins (int): number to bin trial

    Returns:
        np ar: array of binns (i=bin,j=iteration, data=bin count)
    """
    iterations=input_ar.shape[1]
    # y = cluster index
    # x = bin number 1 to 50
    # z = random iteration 1 to 1000
    data_ar=np.zeros(shape=(nr_bins,iterations),dtype=int)
    for it in range(iterations):
        # check if array not empty
        if np.any(input_ar[0,it]):
            data_ar[:,it]=(np.histogram(np.concatenate(input_ar[:,it]).ravel(),bins=nr_bins))[0]
        #if empty
        else:
            data_ar[:,it]=(np.histogram(input_ar[:,it],bins=nr_bins))[0]
    return data_ar




# class ###################################################################################################################
class SpikesSDA():
    def __init__(self, spikes_obj):
        self.session = spikes_obj.session
        self.folder = spikes_obj.folder
        self.gamble_side = spikes_obj.gamble_side

        self.all_trials_df = spikes_obj.all_trials_df
        self.good_trials_df = spikes_obj.good_trials_df
        self.selected_trials_df = spikes_obj.selected_trials_df
        self.skip_clusters = spikes_obj.skip_clusters
        self.spikes_df = spikes_obj.spikes_df
        self.clusters_df = spikes_obj.clusters_df
        
        self.spikes_per_trial_ar = spikes_obj.spikes_per_trial_ar
        self.spikes_per_cluster_ar = spikes_obj.spikes_per_cluster_ar

        #self.randomized_bins_ar = self.get_randomized_samples(200, 1000)



    def get_cluster_name_from_neuron_idx(self, neuron_idx):
        """get the name of the good cluster (global cluster name) from neuron index (position in good cluster)

        Args:
            neuron_idx (int): [description]

        Returns:
            int: [description]
        """
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

 
##STAT ANALYSIS###############################################################################################################

 #Helper Functions statistical data analysis =================================================================================
    def get_bootstrap_all_clusters(self, window, iterations,bins, event):
        """generate data for bootstrap of distribuiton of spikes, to compare event aligend binned spike data
           with iterations time random sampled bins

        Args:
            window (int): 1/2 window width of binned range around event in milli seconds
            iterations (int): number of random samples for each trial
            bins (int): number of bins of each window
            event (str): event to align as comparison

        Returns:
            [type]: [description]
        """
        
        delta=window*20
        
        # initialize data array 
        #y=clusters
        y=self.spikes_per_trial_ar.shape[0]
        #x=trials
        x=self.spikes_per_trial_ar.shape[1]
        #z=random_events 
        z=iterations

        # create zeros data array dtype object
        spiketimes_data_ar = np.zeros(shape=(y,x,z),dtype=object)
        
        # reward alignded database
        reward_aligned_ar = np.zeros((y,x),dtype=object)
        
        #### create random start point array for all trials 
        #get trial data
        trials_ar = np.zeros((x,3))
        trials_ar[:,0]=self.selected_trials_df['start']
        trials_ar[:,1]=self.selected_trials_df['end']
        trials_ar[:,2]=self.selected_trials_df[event]
        #
        random_ar = np.zeros(shape=(x,z),dtype=int)
        random_ar = create_random_start(x,z, trials_ar, delta)

        #get spikes for all clusters
        # get spikes
        for cl in range(y):
            spiketimes_data_ar[cl,:,:] = get_spikes_in_window_all_trial_all_randrang(self.spikes_per_cluster_ar[cl], random_ar, delta)
            reward_aligned_ar[cl,:]=np.array(get_spikes_in_window_all_trials_singlevent(self.spikes_per_cluster_ar[cl], trials_ar[:,2], delta))

        #create flattend spike times for each iteration
        binned = bin_trial_spike_times_all_cluster(spiketimes_data_ar,bins)

        # calulate mean array
        mean_ar = np.mean(binned, axis=2)
        #mean_cl = np.mean(mean_ar, axis=1)
        percentil_ar=np.percentile(binned, [0.5,25,50,75,99.5], axis=2)
        
        return spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar


    def get_all_bootstrap_subselections(self, spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar,bins ):
        # cluster and neuron index
        all_cluster_names = self.clusters_df[self.clusters_df['group']=='good'].index.values
        all_cluster_ids = np.arange(self.spikes_per_cluster_ar.shape[0])


        # selecte rewarded trials
        trial_selector_reward = self.selected_trials_df['reward_given'].values
        trial_selector_no_reward = np.invert(self.selected_trials_df['reward_given'].values)
        trial_selector_gamble = self.selected_trials_df[self.gamble_side].values
        trial_selector_save = np.invert(self.selected_trials_df[self.gamble_side].values)


        # block selector
        blocks = self.selected_trials_df['probability'].unique()

        trial_selector_block1 = self.selected_trials_df['probability']==blocks[0]
        trial_selector_block2 = self.selected_trials_df['probability']==blocks[1]
        trial_selector_block3 = self.selected_trials_df['probability']==blocks[2]

        # Subselected Trials =============================================
        # reward algined subselected reward
        reward_alinged_subselected_reward = reward_aligned_ar[:,trial_selector_reward]
        spiketimes_data_subselected_reward = spiketimes_data_ar[:,trial_selector_reward,:]
        binned_subselected_reward = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_reward,bins)
        mean_subselected_reward = np.mean(binned_subselected_reward, axis=2)
        percentil_subselected_reward = np.percentile(binned_subselected_reward, [0.5,25,50,75,99.5], axis=2)
        rw_dict={
                "reward_alinged":reward_alinged_subselected_reward,
                "spiketimes_data":spiketimes_data_subselected_reward,
                "binned":binned_subselected_reward,
                "mean":mean_subselected_reward,
                "percentiles":percentil_subselected_reward,
                "filename":"reward_aligned_reward"
                }

        # reward aligned subselected no reward
        reward_alinged_subselected_no_reward = reward_aligned_ar[:,trial_selector_no_reward]
        spiketimes_data_subselected_no_reward = spiketimes_data_ar[:,trial_selector_no_reward,:]
        binned_subselected_no_reward = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_no_reward,bins)
        mean_subselected_no_reward = np.mean(binned_subselected_no_reward, axis=2)
        percentil_subselected_no_reward = np.percentile(binned_subselected_no_reward, [0.5,25,50,75,99.5], axis=2)
        norw_dict={
                "reward_alinged":reward_alinged_subselected_no_reward,
                "spiketimes_data":spiketimes_data_subselected_no_reward,
                "binned":binned_subselected_no_reward,
                "mean":mean_subselected_no_reward,
                "percentiles":percentil_subselected_no_reward,
                "filename":"reward_aligned_no_reward"
                }

        # reward aligned subselected gamble
        reward_alinged_subselected_gamble = reward_aligned_ar[:,trial_selector_gamble]
        spiketimes_data_subselected_gamble = spiketimes_data_ar[:,trial_selector_gamble,:]
        binned_subselected_gamble = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_gamble,bins)
        mean_subselected_gamble = np.mean(binned_subselected_gamble, axis=2)
        percentil_subselected_gamble = np.percentile(binned_subselected_gamble, [0.5,25,50,75,99.5], axis=2)
        gamble_dict={
                "reward_alinged":reward_alinged_subselected_gamble,
                "spiketimes_data":spiketimes_data_subselected_gamble,
                "binned":binned_subselected_gamble,
                "mean":mean_subselected_gamble,
                "percentiles":percentil_subselected_gamble,
                "filename":"reward_aligned_gamble"
                }

        # reward aligned subselected safe
        reward_alinged_subselected_save = reward_aligned_ar[:,trial_selector_save]
        spiketimes_data_subselected_save = spiketimes_data_ar[:,trial_selector_save,:]
        binned_subselected_save = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_save,bins)
        mean_subselected_save = np.mean(binned_subselected_save, axis=2)
        percentil_subselected_save = np.percentile(binned_subselected_save, [0.5,25,50,75,99.5], axis=2)
        save_dict={
                "reward_alinged":reward_alinged_subselected_save,
                "spiketimes_data":spiketimes_data_subselected_save,
                "binned":binned_subselected_save,
                "mean":mean_subselected_save,
                "percentiles":percentil_subselected_save,
                "filename":"reward_aligned_save"
                }

        # Blocks Trials ================================================
        # Block 1 reward
        reward_alinged_subselected_block1_rw = reward_aligned_ar[:,( (trial_selector_block1)&(trial_selector_reward) )]
        spiketimes_data_subselected_block1_rw = spiketimes_data_ar[:,( (trial_selector_block1)&(trial_selector_reward) ),:]
        binned_subselected_block1_rw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block1_rw,bins)
        mean_subselected_block1_rw = np.mean(binned_subselected_block1_rw, axis=2)
        percentil_subselected_block1_rw = np.percentile(binned_subselected_block1_rw, [0.5,25,50,75,99.5], axis=2)
        block1_rw_dict={
                "reward_alinged":reward_alinged_subselected_block1_rw,
                "spiketimes_data":spiketimes_data_subselected_block1_rw,
                "binned":binned_subselected_block1_rw,
                "mean":mean_subselected_block1_rw,
                "percentiles":percentil_subselected_block1_rw,
                "filename":"reward_aligned_block_1_reward"
                }

        # Block 1 no-reward
        reward_alinged_subselected_block1_norw = reward_aligned_ar[:,( (trial_selector_block1)&(trial_selector_no_reward) )]
        spiketimes_data_subselected_block1_norw = spiketimes_data_ar[:,( (trial_selector_block1)&(trial_selector_no_reward) ),:]
        binned_subselected_block1_norw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block1_norw,bins)
        mean_subselected_block1_norw = np.mean(binned_subselected_block1_norw, axis=2)
        percentil_subselected_block1_norw = np.percentile(binned_subselected_block1_norw, [0.5,25,50,75,99.5], axis=2)
        block1_norw_dict={
                "reward_alinged":reward_alinged_subselected_block1_norw,
                "spiketimes_data":spiketimes_data_subselected_block1_norw,
                "binned":binned_subselected_block1_norw,
                "mean":mean_subselected_block1_norw,
                "percentiles":percentil_subselected_block1_norw,
                "filename":"reward_aligned_block1_no_reward"
                }

        ##
        # Block 2 reward
        reward_alinged_subselected_block2_rw = reward_aligned_ar[:,( (trial_selector_block2)&(trial_selector_reward) )]
        spiketimes_data_subselected_block2_rw = spiketimes_data_ar[:,( (trial_selector_block2)&(trial_selector_reward) ),:]
        binned_subselected_block2_rw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block2_rw,bins)
        mean_subselected_block2_rw = np.mean(binned_subselected_block2_rw, axis=2)
        percentil_subselected_block2_rw = np.percentile(binned_subselected_block2_rw, [0.5,25,50,75,99.5], axis=2)
        block2_rw_dict={
                "reward_alinged":reward_alinged_subselected_block2_rw,
                "spiketimes_data":spiketimes_data_subselected_block2_rw,
                "binned":binned_subselected_block2_rw,
                "mean":mean_subselected_block2_rw,
                "percentiles":percentil_subselected_block2_rw,
                "filename":"reward_aligned_block2_reward"
                }

        # Bin 2 no-reward
        reward_alinged_subselected_block2_norw = reward_aligned_ar[:,( (trial_selector_block2)&(trial_selector_no_reward) )]
        spiketimes_data_subselected_block2_norw = spiketimes_data_ar[:,( (trial_selector_block2)&(trial_selector_no_reward) ),:]
        binned_subselected_block2_norw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block2_norw,bins)
        mean_subselected_block2_norw = np.mean(binned_subselected_block2_norw, axis=2)
        percentil_subselected_block2_norw = np.percentile(binned_subselected_block2_norw, [0.5,25,50,75,99.5], axis=2)
        block2_norw_dict={
                "reward_alinged":reward_alinged_subselected_block2_norw,
                "spiketimes_data":spiketimes_data_subselected_block2_norw,
                "binned":binned_subselected_block2_norw,
                "mean":mean_subselected_block2_norw,
                "percentiles":percentil_subselected_block2_norw,
                "filename":"reward_aligned_block2_no_reward"
                }

        ##
        # Bin 3 reward
        reward_alinged_subselected_block3_rw = reward_aligned_ar[:,( (trial_selector_block3)&(trial_selector_reward) )]
        spiketimes_data_subselected_block3_rw = spiketimes_data_ar[:,( (trial_selector_block3)&(trial_selector_reward) ),:]
        binned_subselected_block3_rw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block3_rw,bins)
        mean_subselected_block3_rw = np.mean(binned_subselected_block3_rw, axis=2)
        percentil_subselected_block3_rw = np.percentile(binned_subselected_block3_rw, [0.5,25,50,75,99.5], axis=2)
        block3_rw_dict={
                "reward_alinged":reward_alinged_subselected_block3_rw,
                "spiketimes_data":spiketimes_data_subselected_block3_rw,
                "binned":binned_subselected_block3_rw,
                "mean":mean_subselected_block3_rw,
                "percentiles":percentil_subselected_block3_rw,
                "filename":"reward_aligned_block3_reward"
                }

        # Bin 3 no-reward
        reward_alinged_subselected_block3_norw = reward_aligned_ar[:,( (trial_selector_block3)&(trial_selector_no_reward) )]
        spiketimes_data_subselected_block3_norw = spiketimes_data_ar[:,( (trial_selector_block3)&(trial_selector_no_reward) ),:]
        binned_subselected_block3_norw = bin_trial_spike_times_all_cluster(spiketimes_data_subselected_block3_norw,bins)
        mean_subselected_block3_norw = np.mean(binned_subselected_block3_norw, axis=2)
        percentil_subselected_block3_norw = np.percentile(binned_subselected_block3_norw, [0.5,25,50,75,99.5], axis=2)
        block3_norw_dict={
                "reward_alinged":reward_alinged_subselected_block3_norw,
                "spiketimes_data":spiketimes_data_subselected_block3_norw,
                "binned":binned_subselected_block3_norw,
                "mean":mean_subselected_block3_norw,
                "percentiles":percentil_subselected_block3_norw,
                "filename":"reward_aligned_block3_no_reward"
                }

        # generate data dict ================================================
        data_dict = {
                    "all_reward":rw_dict,
                    "all_norw":norw_dict,
                    "all_gamble":gamble_dict,
                    "all_save":save_dict,
                    "block1_rw":block1_rw_dict,
                    "block1_norw":block1_norw_dict,
                    "block2_rw":block2_rw_dict,
                    "block2_norw":block2_norw_dict,
                    "block3_rw":block3_rw_dict,
                    "block3_norw":block3_norw_dict,
                    }

        return data_dict




 #Ploting statistical analysis ============================================================================================
    def plt_surf_single_cluster(self, binned_ar):
        """3D surface + color map plot a surface of bin_counts for given binned data arrayo
        ver bins (x=bin, y=iterations, z=bin_count)

        Args:
            binned_ar (np ar): binned data array for random spike windows selected from all trials
            cluster (int): cluster to plot for

        Returns:
            fix, ax: 
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # get dimension
        # x = bins
        # y = iteration
        # z = spikes in bin
        x,y=binned_ar.shape

        # get data.
        X = np.arange(0,x)
        Y = np.arange(0,y)
        X, Y = np.meshgrid(X, Y)
        # actual data
        Z = binned_ar[:,:].T

        # Plot the surface.
        surf = ax.plot_surface(Y, X, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        return fig, ax

    def plt_test_plot_raw_spikes(self, spikes_ar, binned_ar, cluster, bins):
        """comparison of histograms generated from already binned data 
            and raw spikes for all trials all iterations
            for testing purposis

        Args:
            spikes_ar (np ar): raw spikes for all clusters, trials, and iterations (i=cluster,j=trial,k=iteration,data=np.arry with spike times)
            binned_ar ([type]): already binned data (i=cluster, j=bins, k=iterations, data=count of spikes per bin)
            cluster (int): cluster to plot
            bins (int): number of bins

        Returns:
            fix, ax:
        """
        # create suplots left and right 
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'wspace': 0})
        # iterate over some of the random iterations
        for i in [0,1,5,10,100,500]:
            # create histogram from raw spikes left
            ax[0].hist(np.concatenate(spikes_ar[cluster,:,i]).ravel(),bins=bins)
            # create bar plot from already binned data
            ax[1].bar(np.arange(0,bins),binned_ar[cluster,:,i],width=1.0,label=f"itr:{i}")
        #fix aspect ratio
        [self.fixed_aspect_ratio(0.8,a) for a in ax]
        # create comon legend
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels,loc=8,ncol=6)
        return fig, ax

    def fixed_aspect_ratio(self,ratio,ax):
        """Set a fixed aspect ratio on matplotlib plots 
        regardless of axis units

        Args:
            ratio (foat): x,y ratio
            ax (plt.axs): axis to ratioalize
        """
        xvals,yvals = ax.get_xlim(),ax.get_ylim()

        xrange = xvals[1]-xvals[0]
        yrange = yvals[1]-yvals[0]
        ax.set_aspect(ratio*(xrange/yrange), adjustable='box')

    def plt_compare_random_fixed(self,cluster,window,bins,reward_aligned_ar,mean_ar,percentil_ar):
        """plot summary of randomized bin confidenz interval and event aligned binned spike counts

        Args:
            cluster (int): neuron index
            delta (int): 1/2 window width in samping points
            bins (int): number of binns
            reward_aligned_ar (np ar): spike times for windows event aligned
            mean_ar (np ar): mean spike cont per bin for all random iterations
            percentil_ar (np ar): [0.5,25,50,75,99.5] percentiels spike cont per bin for all random iterations

        Returns:
            [type]: [description]
        """
        delta=window*20
        x=np.linspace(-delta,+delta,bins)
        
        fig,ax = plt.subplots()

        binned_reward = np.histogram(np.concatenate(reward_aligned_ar[cluster,:]).ravel(), bins=bins)[0]
        ax.plot(x,binned_reward, linewidth=3, alpha=1, label="reward aligned")

        # 
        ax.axvline(x=0,linewidth=1, color='r', label="reward event")
        ax.plot(x,mean_ar[cluster], color="black", label="shuffled mean")

        # plot +-95%
        #ax.fill_between(x, np.zeros(bins), percentil_ar[4,:], color='b', alpha=.3, label="0.5th% to 99.5th%")
        ax.fill_between(x, percentil_ar[0,cluster,:], percentil_ar[4,cluster,:], color='b', alpha=0.3, label="0.5th% to 99.5th%")

        
        ax.legend()

        # axis
        labels = [0]
        labels+=np.linspace(-window/1000,window/1000,9,dtype=int).tolist()
        labels.append(0)
        ax.set_xticklabels(labels)
        #labels
        plt.xlabel('window [s]')
        plt.ylabel('spike count')


        #delete
        ax.set_title(f"name:{self.get_cluster_name_from_neuron_idx(cluster)} - idx:{cluster}")
        
        return fig,ax

 # Plot and Save all figures =======================================
    def save_plot(self, name):
        folder = self.folder+"/figures/all_figures"
        plt.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')
        plt.close()
    
    def save_fig(self, name, fig):
        folder = self.folder+"/figures/all_figures"
        fig.savefig(folder+"/"+name+'.png',dpi=200, format='png', bbox_inches='tight')

    def save_all_clusters_compare_random_fixed(self,window,bins,reward_aligned_ar,mean_ar,percentil_ar,name):
        for cluster in range(self.spikes_per_cluster_ar.shape[0]):
            file_name = name+f"_{self.get_cluster_name_from_neuron_idx(cluster)}"
            self.save_fig(file_name,(self.plt_compare_random_fixed(cluster,window,bins,reward_aligned_ar,mean_ar,percentil_ar))[0])


    def generate_plots(self,window,iterations,bins):
        # prepare necessary data arrays
        spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar = self.get_bootstrap_all_clusters(window, iterations, bins, 'reward')
        # get subselections
        data_dict = self.get_all_bootstrap_subselections(spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar,bins)
        
        # iterate over all sub dicts
        for key,value in data_dict.items():
            self.save_all_clusters_compare_random_fixed(window,
                                                        bins,
                                                        value["reward_alinged"],
                                                        value["mean"],
                                                        value["percentiles"],
                                                        value["filename"]
                                                        )
        
