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

def create_random_start(trial_nr,iter_nr, trials_df, delta):
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
    random_li = np.zeros(shape=(trial_nr,iter_nr),dtype=int)
    #iterate over trials
    for index, row in trials_df.iterrows():
        #generate iteration x random event from trial range
        random_li[index,:]=(np.random.randint((row['start']+delta),(row['end']-delta), size=(iter_nr)) )-row['start']
    return random_li


def get_random_range_spikes(data_ar, range_ar):
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
    # iterate over random evets
    for i in range_ar:
        #range_li.append(((i-delta),(i+delta)))
        # get spikes within random event
        results_li.append(data_ar[( (data_ar>=(i-delta)) & (data_ar<=(i+delta)) )]-(i-delta))
        #binned_li.append( (np.histogram(results_li[-1], bins=50, range=[(i-delta),(i+delta)]))[0] )
    return results_li#, binned_li#, range_li #np.array(results_li,dtype=object)


def get_random_range_spikes_all_trials(spikes_per_trial_df, random_ar):
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
        spiketimes_li.append(get_random_range_spikes(spikes_per_trial_df[i].values, random_ar[i]))
        #binnes_li.append(binned_li)
    return spiketimes_li#, binnes_li


def bin_trial_spike_times(input_ar,nr_bins):
    """binn randm windows from all trials all iterations over complete trial

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

        #self.randomized_bins_ar = self.get_randomized_samples(200, 1000)



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

 
##STAT ANALYSIS###############################################################################################################

 #Helper Functions statistical data analysis =================================================================================
    def get_randomized_windows(self, window, iterations):
        """generate data array with random selected events and spike counts for each window around event

        Args:
            window (int): 1/2 window widt in milli seconds
            iterations (int): number of random iterations
        Returns:
            np ar: array with spike counts for i=clusters, j=trials, k=iterations, data = spike times
        """
        # initialize data array 
        #y=clusters
        y=self.spikes_per_trial_ar.shape[0]
        #x=trials
        x=self.spikes_per_trial_ar.shape[1]
        #z=random_events 
        z=iterations

        # translate window from milli seconds to clicks
        delta = window*20

        # create zeros data array dtype object
        data_ar = np.zeros(shape=(y,x,z),dtype=object)

        #### create random start point array for all trials 
        # random ar
        random_ar = np.zeros(shape=(x,z),dtype=int)
        random_ar = create_random_start(x,z, self.selected_trials_df, delta)

        #get spikes for all clusters
        for i in range(y):
            data_ar[i,:,:]=get_random_range_spikes_all_trials(self.spikes_per_trial_ar[i], random_ar)

        return data_ar



 #Ploting statistical analysis ============================================================================================
    def surf_plt(self, binned_ar, cluster):
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
        _,x,y=binned_ar.shape

        # get data.
        X = np.arange(0,x)
        Y = np.arange(0,y)
        X, Y = np.meshgrid(X, Y)
        # actual data
        Z = binned_ar[cluster,:,:].T

        # Plot the surface.
        surf = ax.plot_surface(Y, X, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        return fig, ax

    def test_plot_raw_spikes(self, spikes_ar, binned_ar, cluster, bins):
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
            ax[1].bar(np.arange(0,bins),binned[cluster,:,i],width=1.0,label=f"itr:{i}")
        #fix aspect ratio
        [fixed_aspect_ratio(0.8,a) for a in ax]
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



