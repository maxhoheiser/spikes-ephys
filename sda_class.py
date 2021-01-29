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
        random_li[i,:]=(np.random.randint((trials_ar[i,0]+delta),(trials_ar[i,1]-delta), size=(iter_nr)) )-trials_ar[i,0]
    return random_li

# get spikes for event aligned windows =============================

def get_spikes_in_window_per_trial(data_ar, i, delta):
    """get all spikes that fall into window(+-delta) around event i 

    Args:
        data_ar (np ar): spike times
        i (float): event time
        delta (float): 1/2 window width in sampling points

    Returns:
        np ar: array with spike times that are in window
    """
    return (data_ar[( (data_ar>=(i-delta)) & (data_ar<=(i+delta)) )]-(i-delta))

# fixed event
def get_spikes_in_window_all_trials_singlevent(spikes_ar_all, event_ar, delta):
    """get all spikes that fall in window of specific event for all trials

    Args:
        spikes_ar_all (np ar): array of all spike times for cluster
        event_ar (np ar): array of event times for each trial
        delta (float): 1/2 window width in sampling points

    Returns:
        list: list of arrays with all spike times that fall in window for each trial
    """
    spikes_li_all = list()
    for trial in range(event_ar.shape[0]):
        spikes_li_all.append(get_spikes_in_window_per_trial(spikes_ar_all.values, event_ar[trial], delta))
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
    #binned_li = list()
    #range_li = list()
    for i in range_ar:
        #range_li.append(((i-delta),(i+delta)))
        #results_li.append(data_ar[( (data_ar>=(i-delta)) & (data_ar<=(i+delta)) )]-(i-delta))
        results_li.append(get_spikes_in_window_per_trial(data_ar, i, delta))
        #binned_li.append( (np.histogram(results_li[-1], bins=50, range=[(i-delta),(i+delta)]))[0] )
    return results_li#, binned_li#, range_li #np.array(results_li,dtype=object)


def get_spikes_in_window_all_trial_all_randrang(spikes_per_trial_df, random_ar, delta):
    """get spikes for all trials and iterations

    Args:
        spikes_per_trial_df (np ar): spikes class matrix from
        random_ar (np ar): random events

    Returns:
        li: list of lists with all spikes for all trials and all iterations
    """ 
    spiketimes_li = list()
    #binnes_li = list()
    for i in range(spikes_per_trial_df.shape[0]):
        #results_li, binned_li = get_random_range_spikes(spikes_per_trial_df[i].values, random_ar[i])
        spiketimes_li.append(get_spikes_in_window_per_trial_all_randrang(spikes_per_trial_df[i].values, random_ar[i], delta))
        #binnes_li.append(binned_li)
    return spiketimes_li#, binnes_li


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
        data_ar[:,it]=(np.histogram(np.concatenate(input_ar[:,it]).ravel(),bins=nr_bins))[0]
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
        
        delta = window*20
        
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
        reward_aligned_ar = np.zeros(y,dtype=object)
        
        #### create random start point array for all trials 
        #get trial data
        trial_ar = np.zeros((x,2))
        trial_ar[:,0]=self.selected_trials_df['start']
        trial_ar[:,1]=self.selected_trials_df['end']
        #
        random_ar = np.zeros(shape=(x,z),dtype=int)
        random_ar = create_random_start(x,z, trial_ar, delta)

        #get spikes for all clusters
        for i in range(y):
            spiketimes_data_ar[i,:,:] =get_spikes_in_window_all_trial_all_randrang(self.spikes_per_trial_ar[i], random_ar, delta)

            cluster_name = self.get_cluster_name_from_neuron_idx(i)
            spikes = self.spikes_df[self.spikes_df.loc[:]['cluster'] == cluster_name]['spike_times']
            trials = self.selected_trials_df[event]
            reward_aligned_ar[i]=get_spikes_in_window_all_trials_singlevent(spikes,trials,delta)

        #create flattend spike times for each iteration
        binned = bin_trial_spike_times_all_cluster(spiketimes_data_ar,bins)

        # calulate mean array
        mean_ar = np.mean(binned, axis=2)
        #mean_cl = np.mean(mean_ar, axis=1)
        percentil_ar=np.percentile(binned, [0.5,25,50,75,99.5], axis=2)
        
        return spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar






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

        binned_reward = np.histogram(np.concatenate(reward_aligned_ar[cluster]).ravel(), bins=bins)[0]
        ax.plot(x,binned_reward, linewidth=3, alpha=1, label="reward aligned ")

        # 
        ax.axvline(x=0,linewidth=1, color='r', label="reward")
        ax.plot(x,mean_ar[cluster], color="black", label="mean")

        # plot +-95%
        #ax.fill_between(x, np.zeros(bins), percentil_ar[4,:], color='b', alpha=.3, label="0.5th% to 99.5th%")
        ax.fill_between(x, percentil_ar[0,cluster,:], percentil_ar[4,cluster,:], color='b', alpha=0.3, label="0.5th% to 99.5th%")

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels,loc=1,ncol=6)
        
        return fig,ax

