import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
mpl.style.use('seaborn')
plt.rcParams['axes.facecolor'] = '#f0f4f7'
plt.rc('legend', frameon=True,fancybox=True, framealpha=1)
blue = '#4C72B0'
green = '#55A868'
red = '#C44E52'
purple = '#8172B2'
yellow = '#CCB974'
lightblue = '#64B5CD'

import csv
import scipy.stats as st
import platform
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
import copy

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from scipy.stats import shapiro

# for pylatex
from pylatex import (
    Document,
    Section,
    Subsection,
    Command,
    Package,
    NewPage,
    LongTabu,
    Tabular,
)
from pylatex.utils import italic, NoEscape

# numba helper functions
from numba import njit

default = [6.4, 4.8]


@njit()
def create_random_start(trial_nr, iter_nr, trials_ar, delta):
    """get random event within trial

    Args:
        trial_nr (int): number of trials
        iter_nr (int): number of iterations
        trials_df (numpy ar): dataframe from spikes class with all trials
        delta (float): window = 2*delta

    Returns:
        random_li(numpy ar): array with random start points, (i=trial_nr, j=iter_nr)
    """
    # initialize complete dataframe
    random_li = np.zeros(shape=(trial_nr, iter_nr))
    # iterate over trials
    for i in range(trial_nr):
        random_li[i, :] = np.random.randint(
            (trials_ar[i, 0]), (trials_ar[i, 1]), size=(iter_nr)
        )
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
    return (data_ar[((data_ar >= (i - delta)) & (data_ar <= (i + delta)))]) - (i)


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
    # binnes_li = list()
    for i in range(random_ar.shape[0]):
        spiketimes_li.append(
            get_spikes_in_window_per_trial_all_randrang(spikes_ar, random_ar[i], delta)
        )
    return spiketimes_li


# binn data
def bin_trial_spike_times_all_cluster(input_ar, nr_bins):
    """binn randm windows from all clusters, all trials all iterations over complete trial

    Args:
        input_ar (np ar): spikes per random event for all clusters, all trials, all iterations
        nr_bins (int): number to bin trial

    Returns:
        np ar: array of binns (i=cluster,j=bin,k=iteration, data=bin count)
    """
    cluster = input_ar.shape[0]
    iterations = input_ar.shape[2]
    # y = cluster index
    # x = bin number 1 to 50
    # z = random iteration 1 to 1000
    data_ar = np.zeros(shape=(cluster, nr_bins, iterations), dtype=int)
    for cl in range(cluster):
        for it in range(iterations):
            data_ar[cl, :, it] = get_histogram((input_ar[cl, :, it]), bins=nr_bins)
    return data_ar


def get_histogram(data, bins):
    try:
        hist = np.histogram(np.concatenate(data).ravel(), bins=bins)[0]
    except:
        hist = np.histogram((data), bins=bins)[0]
    return hist


def bin_trial_spike_times_single_cluster(input_ar, nr_bins):
    """binn randm windows from single clusters, all trials all iterations over complete trial

    Args:
        input_ar (np ar): spikes per random event for single clusters, all trials, all iterations
        nr_bins (int): number to bin trial

    Returns:
        np ar: array of binns (i=bin,j=iteration, data=bin count)
    """
    iterations = input_ar.shape[1]
    # y = cluster index
    # x = bin number 1 to 50
    # z = random iteration 1 to 1000
    data_ar = np.zeros(shape=(nr_bins, iterations), dtype=int)
    for it in range(iterations):
        # check if array not empty
        data_ar[:, it] = get_histogram((input_ar[:, it]), bins=nr_bins)
    return data_ar


# class ###################################################################################################################
class SpikesSDA:
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

        self.blocks = self.selected_trials_df["probability"].unique()

    def get_cluster_name_from_neuron_idx(self, neuron_idx):
        """get the name of the good cluster (global cluster name) from neuron index (position in good cluster)

        Args:
            neuron_idx (int): [description]

        Returns:
            int: [description]
        """
        cluster_name = (
            self.clusters_df.loc[self.clusters_df["group"] == "good"]
            .iloc[neuron_idx]
            .name
        )
        return cluster_name

    def get_neuron_idx_from_cluster_name(self, cluster_name):
        """return the index of cluster name in only good neurons -> find in spikes_per_trial_ar
        Args:
            cluster_name (int): original index of good cluster in clusters_df 
        Returns:
            int: index of cluster in spikes_per_trial_ar
        """
        neuron_idx = (
            np.where(
                self.clusters_df.loc[self.clusters_df["group"] == "good"].index.values
                == cluster_name
            )
        )[0][0]
        return neuron_idx

    def load_bootrstp(self, window, iterations, bins):
        self.bins = bins
        self.window = window
        self.iterations = iterations
        (
            self.spiketimes_data_ar,
            self.reward_aligned_ar,
            self.binned,
            self.mean_ar,
            self.percentil_ar,
        ) = self.get_bootstrap_all_clusters(window, iterations, bins, "reward")

    def load_data_dict(self, window, iterations, bins, reload=False):
        # check if first bootrstrap in locals:
        try:
            self.window
            self.iterations
            self.bins
            self.spiketimes_data_ar
            self.reward_aligned_ar
            self.binned
            self.mean_ar
            self.percentil_ar
        except:
            print("not intial bootstrap -> load it")
            self.load_bootrstp(window, iterations, bins)
        else:
            print("there")
            pass

        # prepare data
        try:
            self.data_dict
        except:
            print("no data dict -> load it")
            self.data_dict = self.get_all_bootstrap_subselections(
                self.spiketimes_data_ar,
                self.reward_aligned_ar,
                bins,
                self.binned,
                self.mean_ar,
                self.percentil_ar,
            )
        else:
            if reload:
                print("reload")
                self.data_dict = self.get_all_bootstrap_subselections(
                    self.spiketimes_data_ar,
                    self.reward_aligned_ar,
                    bins,
                    self.binned,
                    self.mean_ar,
                    self.percentil_ar,
                )
        return self.data_dict

    def load_info_df(self, sig_number=2):
        self.info_df = self.add_session_session_sig_info(
            data_dict=self.data_dict, bins=self.bins, sig_number=sig_number
        )
        return self.info_df

    ##STAT ANALYSIS###############################################################################################################

    # Helper Functions statistical data analysis =================================================================================
    def get_bootstrap_all_clusters(self, window, iterations, bins, event):
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

        delta = window * 20

        # initialize data array
        # y=clusters
        y = self.spikes_per_trial_ar.shape[0]
        # x=trials
        x = self.spikes_per_trial_ar.shape[1]
        # z=random_events
        z = iterations

        # create zeros data array dtype object
        spiketimes_data_ar = np.zeros(shape=(y, x, z), dtype=object)

        # reward alignded database
        reward_aligned_ar = np.zeros((y, x), dtype=object)

        #### create random start point array for all trials
        # get trial data
        trials_ar = np.zeros((x, 3))
        trials_ar[:, 0] = self.selected_trials_df["start"]
        trials_ar[:, 1] = self.selected_trials_df["end"]
        trials_ar[:, 2] = self.selected_trials_df[event]
        #
        random_ar = np.zeros(shape=(x, z), dtype=int)
        random_ar = create_random_start(x, z, trials_ar, delta)

        # get spikes for all clusters
        # get spikes
        for cl in range(y):
            spiketimes_data_ar[cl, :, :] = get_spikes_in_window_all_trial_all_randrang(
                self.spikes_per_cluster_ar[cl], random_ar, delta
            )
            reward_aligned_ar[cl, :] = np.array(
                get_spikes_in_window_all_trials_singlevent(
                    self.spikes_per_cluster_ar[cl], trials_ar[:, 2], delta
                )
            )

        # create flattend spike times for each iteration
        binned = bin_trial_spike_times_all_cluster(spiketimes_data_ar, bins)

        # calulate mean array
        mean_ar = np.mean(binned, axis=2)
        # mean_cl = np.mean(mean_ar, axis=1)
        percentil_ar = np.percentile(binned, [0.5, 25, 50, 75, 99.5], axis=2)

        return spiketimes_data_ar, reward_aligned_ar, binned, mean_ar, percentil_ar

    def get_bootstrap_subselection_dict(
        self, trial_selector, filename, bins, reward_aligned_ar, spiketimes_data_ar
    ):
        reward_alinged_subselected = reward_aligned_ar[:, trial_selector]
        spiketimes_data_subselected = spiketimes_data_ar[:, trial_selector, :]
        binned_subselected = bin_trial_spike_times_all_cluster(
            spiketimes_data_subselected, bins
        )
        mean_subselected = np.mean(binned_subselected, axis=2)
        percentil_subselected = np.percentile(
            binned_subselected, [0.5, 25, 50, 75, 99.5], axis=2
        )
        # get fingerprint
        fingerprint_per = self.get_fingerprint(
            reward_alinged_subselected,
            percentil_subselected[0],
            percentil_subselected[4],
            bins,
        )
        var_ar = np.var(binned_subselected, axis=2)
        mean_ar = np.mean(binned_subselected, axis=2)
        fingerprint_sig = self.get_fingerprint(
            reward_alinged_subselected, mean_ar - 2 * var_ar, mean_ar + 2 * var_ar, bins
        )

        dict = {
            "reward_alinged": reward_alinged_subselected,
            "spiketimes_data": spiketimes_data_subselected,
            "binned": binned_subselected,
            "mean": mean_subselected,
            "percentiles": percentil_subselected,
            "filename": filename,
            "fingerprint_sig": fingerprint_sig,
            "fingerprint_per": fingerprint_per,
        }
        return dict

    def get_all_bootstrap_subselections(
        self,
        spiketimes_data_ar,
        reward_aligned_ar,
        bins,
        binned_ar,
        mean_ar,
        percentil_ar,
    ):
        # cluster and neuron index
        all_cluster_names = self.clusters_df[
            self.clusters_df["group"] == "good"
        ].index.values
        all_cluster_ids = np.arange(self.spikes_per_cluster_ar.shape[0])

        # selecte rewarded trials
        trial_selector_reward = self.selected_trials_df["reward_given"].values
        trial_selector_no_reward = np.invert(
            self.selected_trials_df["reward_given"].values
        )
        trial_selector_gamble = self.selected_trials_df[self.gamble_side].values
        trial_selector_safe = np.invert(
            self.selected_trials_df[self.gamble_side].values
        )

        # block selector
        blocks = self.selected_trials_df["probability"].unique()

        trial_selector_block1 = self.selected_trials_df["probability"] == blocks[0]
        trial_selector_block2 = self.selected_trials_df["probability"] == blocks[1]
        trial_selector_block3 = self.selected_trials_df["probability"] == blocks[2]

        # all trials
        fingerprint_per = self.get_fingerprint(
            reward_aligned_ar, percentil_ar[0], percentil_ar[4], bins
        )
        var_ar = np.var(binned_ar, axis=2)
        mean_ar = np.mean(binned_ar, axis=2)
        fingerprint_sig = self.get_fingerprint(
            reward_aligned_ar, mean_ar - 1 * var_ar, mean_ar + 1 * var_ar, bins
        )
        all_dict = {
            "reward_alinged": reward_aligned_ar,
            "spiketimes_data": spiketimes_data_ar,
            "binned": binned_ar,
            "mean": mean_ar,
            "percentiles": percentil_ar,
            "filename": "reward_aligned_block3_no_reward",
            "fingerprint_sig": fingerprint_sig,
            "fingerprint_per": fingerprint_per,
        }

        # generate data dict ================================================
        # get_bootstrap_subselection_dict(trial_selector,
        #                                 filename,
        #
        #                                 bins,
        #                                 reward_aligned_ar,
        #                                 spiketimes_data_ar)

        data_dict = {
            "all": all_dict,
            "rw": self.get_bootstrap_subselection_dict(
                trial_selector_reward,
                "reward_aligned_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "norw": self.get_bootstrap_subselection_dict(
                trial_selector_no_reward,
                "reward_aligned_no_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "gamble": self.get_bootstrap_subselection_dict(
                trial_selector_gamble,
                "reward_aligned_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "safe": self.get_bootstrap_subselection_dict(
                trial_selector_safe,
                "reward_aligned_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "rw_gamble": self.get_bootstrap_subselection_dict(
                ((trial_selector_gamble) & (trial_selector_reward)),
                "reward_aligned_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "norw_gamble": self.get_bootstrap_subselection_dict(
                ((trial_selector_gamble) & (trial_selector_no_reward)),
                "reward_aligned_no_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "rw_safe": self.get_bootstrap_subselection_dict(
                ((trial_selector_safe) & (trial_selector_reward)),
                "reward_aligned_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "norw_safe": self.get_bootstrap_subselection_dict(
                ((trial_selector_safe) & (trial_selector_no_reward)),
                "reward_aligned_no_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            # block 1
            "block1_all": self.get_bootstrap_subselection_dict(
                trial_selector_block1,
                "reward_aligned_block1_all",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_rw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block1) & (trial_selector_reward)),
                "reward_aligned_block1_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_norw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block1) & (trial_selector_no_reward)),
                "reward_aligned_block1_no_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_gamble": self.get_bootstrap_subselection_dict(
                ((trial_selector_block1) & (trial_selector_gamble)),
                "reward_aligned_block1_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_rw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block1)
                    & (trial_selector_gamble)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block1_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_norw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block1)
                    & (trial_selector_gamble)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block1_no_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_safe": self.get_bootstrap_subselection_dict(
                ((trial_selector_block1) & (trial_selector_safe)),
                "reward_aligned_block1_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_rw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block1)
                    & (trial_selector_safe)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block1_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block1_norw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block1)
                    & (trial_selector_safe)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block1_no_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            # block 2
            "block2_all": self.get_bootstrap_subselection_dict(
                trial_selector_block2,
                "reward_aligned_block2_all",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_rw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block2) & (trial_selector_reward)),
                "reward_aligned_block2_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_norw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block2) & (trial_selector_no_reward)),
                "reward_aligned_block2_no_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_gamble": self.get_bootstrap_subselection_dict(
                ((trial_selector_block2) & (trial_selector_gamble)),
                "reward_aligned_block2_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_rw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block2)
                    & (trial_selector_gamble)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block2_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_norw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block2)
                    & (trial_selector_gamble)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block2_no_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_safe": self.get_bootstrap_subselection_dict(
                ((trial_selector_block2) & (trial_selector_safe)),
                "reward_aligned_block2_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_rw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block2)
                    & (trial_selector_safe)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block2_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block2_norw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block2)
                    & (trial_selector_safe)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block2_no_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            # block 3
            "block3_all": self.get_bootstrap_subselection_dict(
                trial_selector_block3,
                "reward_aligned_block3_all",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_rw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block3) & (trial_selector_reward)),
                "reward_aligned_block3_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_norw": self.get_bootstrap_subselection_dict(
                ((trial_selector_block3) & (trial_selector_no_reward)),
                "reward_aligned_block3_no_reward",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_gamble": self.get_bootstrap_subselection_dict(
                ((trial_selector_block3) & (trial_selector_gamble)),
                "reward_aligned_block3_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_rw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block3)
                    & (trial_selector_gamble)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block3_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_norw_gamble": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block3)
                    & (trial_selector_gamble)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block3_no_reward_gamble",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_safe": self.get_bootstrap_subselection_dict(
                ((trial_selector_block3) & (trial_selector_safe)),
                "reward_aligned_block3_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_rw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block3)
                    & (trial_selector_safe)
                    & (trial_selector_reward)
                ),
                "reward_aligned_block3_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
            "block3_norw_safe": self.get_bootstrap_subselection_dict(
                (
                    (trial_selector_block3)
                    & (trial_selector_safe)
                    & (trial_selector_no_reward)
                ),
                "reward_aligned_block3_no_reward_safe",
                bins,
                reward_aligned_ar,
                spiketimes_data_ar,
            ),
        }

        return data_dict

    def scale_data(self, data):
        if data.ndim == 2:
            # for 2d array i=samples, j=features
            scaler = StandardScaler()
            scaler.fit(data)
            data_scaled = scaler.transform(data)

        if data.ndim == 1:
            # for 1d array
            data_scaled = scale(data)
        else:
            data_scaled = None

        return data_scaled

    def shapiro_wilk_test(self, data):
        # normality test
        stat, p = shapiro(data)
        print("Statistics=%.3f, p=%.3f" % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print("Sample looks Gaussian (fail to reject H0)")
        else:
            print("Sample does not look Gaussian (reject H0)")
        return stat, p

    def get_fingerprint(self, reward_alinged_ar, lower_ar, upper_ar, bins):
        # get bins that are outside of threshold +/- for each neuron
        # parameter
        all_clusters_names = self.clusters_df[
            self.clusters_df["group"] == "good"
        ].index.values
        nr_clusters = all_clusters_names.shape[0]
        # -1 under, 0 in, +1 above
        fingerprint_ar = np.zeros((nr_clusters, bins), dtype=int)

        for cl in range(nr_clusters):
            bins_aligned = get_histogram((reward_alinged_ar[cl]), bins)
            fingerprint_ar[cl][bins_aligned > upper_ar[cl]] = int(1)
            fingerprint_ar[cl][bins_aligned < lower_ar[cl]] = int(-1)

        # dataframe from array
        # neurons_fingerprint_df=pd.DataFrame(all_clusters_names,columns="neuron name")
        # columns=[f"bin {bin}" for bin in range(bins)]
        # neurons_fingerprint_df[columns]=neurons_fingerprint
        fingerprint_df = pd.DataFrame(
            fingerprint_ar, columns=[f"bin {bin}" for bin in range(bins)]
        )
        fingerprint_df["below"] = (fingerprint_df == int(-1)).sum(axis=1)
        fingerprint_df["in"] = (fingerprint_df == int(0)).sum(axis=1)
        fingerprint_df["above"] = (fingerprint_df == int(1)).sum(axis=1)

        return fingerprint_df

    def create_all_sessions_info_df(self):
        fixed_columns = [
            "session",
            "tot. clusters",
            "nr. good",
            "nr. mua",
            "nr. noise",
            "tot. trials",
            "good trials",
            "selected trials",
            "rw block 1",
            "rw block 2",
            "rw block 3",
            "len block 1",
            "len block 2",
            "len block 3",
        ]
        # columns_all =========================##########################==================================
        trials_list = [
            "all",
            "rw",
            "norw",
            "gamble",
            "rw_gamble",
            "norw_gamble",
            "safe",
            "rw_safe",
            "norw_safe",
        ]

        columns_all = list()
        for trials in trials_list:
            columns_all.append(f"{trials} before")
            columns_all.append(f"{trials} before neurons")
            columns_all.append(f"{trials} across")
            columns_all.append(f"{trials} across neurons")
            columns_all.append(f"{trials} after")
            columns_all.append(f"{trials} after neurons")
            columns_all.append(f"{trials} all")
            columns_all.append(f"{trials} all neurons")
        # columns_blocks
        columns_blocks = list()
        for block in [1, 2, 3]:
            # iterate over all , rw, norw
            for trials in trials_list:
                columns_blocks.append(f"{block} {trials} before")
                columns_blocks.append(f"{block} {trials} before neurons")
                columns_blocks.append(f"{block} {trials} across")
                columns_blocks.append(f"{block} {trials} across neurons")
                columns_blocks.append(f"{block} {trials} after")
                columns_blocks.append(f"{block} {trials} after neurons")
                columns_blocks.append(f"{block} {trials} all")
                columns_blocks.append(f"{block} {trials} all neurons")

        info_df = pd.DataFrame(columns=(fixed_columns + columns_all + columns_blocks))
        return info_df

    def add_session_session_sig_info(
        self, data_dict, bins, info_df=False, sig_number=4
    ):
        if info_df == False:
            info_df = self.create_all_sessions_info_df()
        current_index = info_df.shape[0] + 1
        info_df.loc[current_index, "session"] = self.session

        # columns_all =========================##########################==================================
        trials_list = [
            "all",
            "rw",
            "norw",
            "gamble",
            "rw_gamble",
            "norw_gamble",
            "safe",
            "rw_safe",
            "norw_safe",
        ]

        cluster_count = self.clusters_df["group"].value_counts().values
        info_df.loc[
            info_df["session"] == self.session,
            ["tot. clusters", "nr. good", "nr. mua", "nr. noise"],
        ] = [cluster_count.sum()] + cluster_count.tolist()
        # trials info
        info_df.loc[
            info_df["session"] == self.session,
            ["tot. trials", "good trials", "selected trials"],
        ] = [
            self.all_trials_df.shape[0],
            self.good_trials_df.shape[0],
            self.selected_trials_df.shape[0],
        ]
        # block info
        blocks = self.selected_trials_df["probability"].unique()
        info_df.loc[
            info_df["session"] == self.session,
            ["rw block 1", "rw block 2", "rw block 3"],
        ] = blocks
        info_df.loc[info_df["session"] == self.session, ["len block 1"]] = (
            self.selected_trials_df[self.selected_trials_df["probability"] == blocks[0]]
        ).shape[0]
        info_df.loc[info_df["session"] == self.session, ["len block 2"]] = (
            self.selected_trials_df[self.selected_trials_df["probability"] == blocks[1]]
        ).shape[0]
        info_df.loc[info_df["session"] == self.session, ["len block 3"]] = (
            self.selected_trials_df[self.selected_trials_df["probability"] == blocks[2]]
        ).shape[0]
        # neural findings===============================================================================================
        # neural response all trials========================================
        for trials in trials_list:
            key = f"{trials}"
            # data = data_dict[key]["reward_alinged"]
            # lower = data_dict[key]["percentiles"][0]
            # upper = data_dict[key]["percentiles"][4]
            # fingerprint = self.get_fingerprint(data,lower,upper,bins)
            fingerprint = data_dict[key]["fingerprint_per"]
            # before reward event [tot number, indeces]
            before = np.where(
                (fingerprint.loc[:, "bin 20":"bin 25"] > 0).sum(axis=1) >= sig_number
            )[0]
            across = np.where(
                (fingerprint.loc[:, "bin 23":"bin 27"] > 0).sum(axis=1) >= sig_number
            )[0]
            after = np.where(
                (fingerprint.loc[:, "bin 25":"bin 30"] > 0).sum(axis=1) >= sig_number
            )[0]
            # get intersecting values
            true_before = np.array(
                [i for i in before if i not in np.concatenate((across, after))]
            )
            true_after = np.array(
                [i for i in after if i not in np.concatenate((before, across))]
            )
            all_unique = np.unique(np.concatenate([before, across, after], axis=0))
            # add to dataframe
            info_df.at[current_index, f"{trials} before"] = true_before.shape[0]
            info_df.at[current_index, f"{trials} before neurons"] = true_before.tolist()
            # across reward event [tot number, indeces]
            info_df.at[current_index, f"{trials} across"] = across.shape[0]
            info_df.at[current_index, f"{trials} across neurons"] = across.tolist()
            # after reward event [tot number, indeces]
            info_df.at[current_index, f"{trials} after"] = true_after.shape[0]
            info_df.at[current_index, f"{trials} after neurons"] = true_after.tolist()
            # all unique before, across and after
            info_df.at[current_index, f"{trials} all"] = all_unique.shape[0]
            info_df.at[current_index, f"{trials} all neurons"] = all_unique.tolist()

        # nerual response blocks ========================================
        for block in [1, 2, 3]:
            # iterate over all , rw, norw
            for trials in trials_list:
                # all
                key = f"block{block}_{trials}"
                # data = data_dict[key]["reward_alinged"]
                # lower = data_dict[key]["percentiles"][0]
                # upper = data_dict[key]["percentiles"][4]
                # fingerprint = self.get_fingerprint(data,lower,upper,bins)
                fingerprint = data_dict[key]["fingerprint_per"]
                # get values
                before = np.where(
                    (fingerprint.loc[:, "bin 20":"bin 25"] > 0).sum(axis=1)
                    >= sig_number
                )[0]
                across = np.where(
                    (fingerprint.loc[:, "bin 23":"bin 27"] > 0).sum(axis=1)
                    >= sig_number
                )[0]
                after = np.where(
                    (fingerprint.loc[:, "bin 25":"bin 30"] > 0).sum(axis=1)
                    >= sig_number
                )[0]
                # get intersecting values
                true_before = np.array(
                    [i for i in before if i not in np.concatenate((across, after))]
                )
                true_after = np.array(
                    [i for i in after if i not in np.concatenate((before, across))]
                )
                all_unique = np.unique(np.concatenate([before, across, after], axis=0))
                # before reward event [tot number, indeces]
                info_df.at[1, f"{block} {trials} before"] = true_before.shape[0]
                info_df.at[1, f"{block} {trials} before neurons"] = before.tolist()
                # across reward event [tot number, indeces]
                info_df.at[1, f"{block} {trials} across"] = across.shape[0]
                info_df.at[1, f"{block} {trials} across neurons"] = across.tolist()
                # after reward event [tot number, indeces]
                info_df.at[1, f"{block} {trials} after"] = true_after.shape[0]
                info_df.at[1, f"{block} {trials} after neurons"] = after.tolist()
                # all unique neurons before, across or after
                info_df.at[current_index, f"{block} {trials} all"] = all_unique.shape[0]
                info_df.at[
                    current_index, f"{block} {trials} all neurons"
                ] = all_unique.tolist()

        return info_df

    # Ploting statistical analysis ======s======================================================================================
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
        ax = fig.gca(projection="3d")

        # get dimension
        # x = bins
        # y = iteration
        # z = spikes in bin
        x, y = binned_ar.shape

        # get data.
        X = np.arange(0, x)
        Y = np.arange(0, y)
        X, Y = np.meshgrid(X, Y)
        # actual data
        Z = binned_ar[:, :].T

        # Plot the surface.
        surf = ax.plot_surface(Y, X, Z, linewidth=0, antialiased=False)

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
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={"wspace": 0})
        # iterate over some of the random iterations
        for i in [0, 1, 5, 10, 100, 500]:
            # create histogram from raw spikes left
            try:
                ax[0].hist(np.concatenate(spikes_ar[cluster, :, i]).ravel(), bins=bins)
            except:
                ax[0].hist((spikes_ar[cluster, :, i]), bins=bins)
            # create bar plot from already binned data
            ax[1].bar(
                np.arange(0, bins),
                binned_ar[cluster, :, i],
                width=1.0,
                label=f"itr:{i}",
            )
        # fix aspect ratio
        [self.fixed_aspect_ratio(0.8, a) for a in ax]
        # create comon legend
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc=8, ncol=6)
        return fig, ax

    def fixed_aspect_ratio(self, ratio, ax):
        """Set a fixed aspect ratio on matplotlib plots 
        regardless of axis units

        Args:
            ratio (foat): x,y ratio
            ax (plt.axs): axis to ratioalize
        """
        xvals, yvals = ax.get_xlim(), ax.get_ylim()

        xrange = xvals[1] - xvals[0]
        yrange = yvals[1] - yvals[0]
        ax.set_aspect(ratio * (xrange / yrange), adjustable="box")

    def plt_compare_random_fixed(
        self, cluster, window, bins, reward_aligned_ar, mean_ar, percentil_ar,figsize=default
    ):
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
        delta = window * 20
        x = np.linspace(-delta, +delta, bins)

        fig, ax = plt.subplots(1,1,figsize=figsize)

        binned_reward = get_histogram((reward_aligned_ar[cluster, :]), bins)
        ax.plot(x, binned_reward, linewidth=2, alpha=1, label="event aligned", color=blue)
        #
        ax.axvline(x=0, linewidth=1, color=red, label="event")
        ax.plot(x, mean_ar[cluster], color="black", label="shuffled mean", linewidth=1, linestyle='--')
        # plot +-95%
        # ax.fill_between(x, np.zeros(bins), percentil_ar[4,:], color='b', alpha=.3, label="0.5th% to 99.5th%")
        ax.fill_between(
            x,
            percentil_ar[0, cluster, :],
            percentil_ar[4, cluster, :],
            color='blue',
            alpha=0.3,
            label="0.5th% to 99.5th%",
            rasterized=True
        )

        ax.legend()
        # axis
        labels = [0]
        labels += np.linspace(-window / 1000, window / 1000, 5, dtype=int).tolist()
        labels.append(0)
        ax.set_xticklabels(labels)
        # labels
        plt.xlabel("window [s]")
        plt.ylabel("spike count")
        # delete
        ax.set_title(
            f"name:{self.get_cluster_name_from_neuron_idx(cluster)} - idx:{cluster}"
        )

        return fig, ax

    def plt_compare_random_fixed_sigma(
        self, cluster, window, bins, reward_aligned_ar, mean_ar, sigma_ar,figsize=default
    ):
        delta = window * 20
        x = np.linspace(-delta, +delta, bins)

        fig, ax = plt.subplots(1,1,figsize=figsize)

        binned_reward = get_histogram((reward_aligned_ar[cluster, :]), bins=bins)
        ax.plot(x, binned_reward, linewidth=2, alpha=1, label="event aligned")
        #
        ax.axvline(x=0, linewidth=1, color=red, label="event")
        ax.plot(x, mean_ar[cluster], color="black", label="shuffled mean", linewidth=1, linestyle='--')

        # plot +-95%
        # ax.fill_between(x, np.zeros(bins), percentil_ar[4,:], color='b', alpha=.3, label="0.5th% to 99.5th%")
        # +-1sigma
        for factor in [1, 2, 3]:
            low = mean_ar[cluster] - factor * sigma_ar[cluster]
            high = mean_ar[cluster] + factor * sigma_ar[cluster]
            ax.fill_between(
                x,
                low,
                high,
                color='blue',
                alpha=0.2 * (1 / factor),
                label=f"+-{factor}sigma",
                rasterized=True
            )
        # -2sigma
        # -2sigma
        ax.legend()
        # axis
        labels = [0]
        labels += np.linspace(-window / 1000, window / 1000, 5, dtype=int).tolist()
        labels.append(0)
        ax.set_xticklabels(labels)
        # labels
        plt.xlabel("window [s]")
        plt.ylabel("spike count")
        # delete
        # set x range
        ax.set_xlim([-window/1000,window/1000])
        # set title
        ax.set_title(
            f"name:{self.get_cluster_name_from_neuron_idx(cluster)} - idx:{cluster}"
        )

        return fig, ax

    def plt_fit_normdist(self, data, figsize=default):
        """plot normaldistributin fitted to histogram

        Args:
            data (np ar): input data[samples,features]
        """
        mu, std = norm.fit(data)

        fig, ax = plt.subplots(1,1, figsize=figsize)
        # plot histogram
        ax.hist(data, bins=25, density=True, alpha=0.6, color=blue, label="bin count",rasterized=True)

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, "k", linewidth=2, label="normal fit\nmu:%.2f\nstd:%.2f"% (mu, std)) 
        #title = "Normal distribution fitted to data \n(mu:%.2f, std:%.2f)" % (mu, std)
        # namings usw
        ax.set_xlabel("bin count")
        ax.set_ylabel("probability")
        ax.legend()
        #ax.set_title(title)

        return fig,ax

    def colormap(self,info_df,figsize=default):
        x_labels = ['tot. trials','reward','no-reward','gamble', 'safe','gamble reward','safe reward', 'gamble no-reward', 'safe no-reward']
        y_labels = ['all trials','75% block', "25% block", "12.5% block"]


        data = np.zeros([4,9],dtype=int)

        data[0,:] = info_df.loc[:,['all all','rw all','norw all','gamble all', 'safe all','rw_gamble all','rw_safe all', 'norw_gamble all', 'norw_safe all']].values
        for block in [1,2,3]:
            data[block,:] = info_df.loc[:,[f'{block} all all',
                                        f'{block} rw all',
                                        f'{block} norw all',
                                        f'{block} gamble all', 
                                        f'{block} safe all',
                                        f'{block} rw_gamble all',
                                        f'{block} rw_safe all', 
                                        f'{block} norw_gamble all', 
                                        f'{block} norw_safe all']
                                    ].values


        #data = data.astype(int)

        fig, ax = plt.subplots(1,1,figsize=figsize)
        #im = ax.imshow(data,norm=LogNorm())

        my_cmap = copy.copy(mpl.cm.get_cmap('viridis')) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        im = ax.imshow(data, 
                norm=mpl.colors.LogNorm(), 
                interpolation='nearest', 
                cmap=my_cmap,
                rasterized=True
                )

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

    def plt_fingerprint_2d(
        self, fingerprint_df, axis_label, fig=None, ax=None, title=None, figsize=default
    ):

        if fig == None and ax == None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            no_details = True
        else:
            no_details = False

        test_x = fingerprint_df["below"]
        test_y = fingerprint_df["above"]

        # Generate a list of unique points
        points = list(set(zip(test_x, test_y)))
        # Generate a list of point counts
        count = [
            len([x for x, y in zip(test_x, test_y) if x == p[0] and y == p[1]])
            for p in points
        ]
        # Now for the plotting:
        plot_x = [i[0] for i in points]
        plot_y = [i[1] for i in points]
        count = np.array(count)
        im = ax.scatter(
            plot_x, plot_y, c=count, s=60 * count ** 0.95, cmap="Spectral_r", alpha=0.8
        )

        if no_details:
            cbr= fig.colorbar(im, orientation="vertical",)
            cbr.ax.set_yticklabels(np.linspace(1,count.max()+2,9).astype(int))

        ax.scatter(plot_x, plot_y, c="black", s=2)
        ax.set_xlabel(f"below {axis_label[1]}")
        ax.set_ylabel(f"above {axis_label[0]}")

        ax.set_yticks(np.arange(0, (test_y.max() + 1)))

        if title != None:
            ax.set_title(title)

        return fig, ax, im

    def plt_neuron_fingerprint_all(self, data, title, fig=None, ax=None, ylabel=False, figsize=default):
        if fig == None and ax == None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            no_details = True
        else:
            no_details = False

        # make sure that in data is -1,and +1 -> assign 0.0 = -1 and 0.1 = +1
        data[0, 0] = -1
        data[0, 1] = +1
        # define colormap
        cmap = plt.get_cmap("viridis", np.max(data) - np.min(data) + 1)
        # plot data
        im = ax.pcolor(data, cmap=cmap,rasterized=True)
        # plot reward line
        ax.axvline(25, color="red", label="event")

        if no_details:
            # tell the colorbar to tick at integers
            cbar = fig.colorbar(im, ticks=np.arange(np.min(data), np.max(data) + 1))
            cbar.ax.set_yticklabels(
                ["below", "in", "above"]
            )  # vertically oriented colorbar
            # add legend
            ax.legend()
        # x y labels
        ax.set_xlabel("bin")
        if ylabel == False:
            ax.set_ylabel("Neuron")
        else:
            ax.set_ylabel(ylabel)

        # set text
        if title != False:
            ax.set_title(f"{title}")

        fig.tight_layout()

        return fig, ax, im

    def autolabel(self, rects, ax):
        # Attach a text label above each bar in *rects*, displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def plt_bar_sig_neurons(self, row, trials, title=False, fig=None, ax=None, x_label=False, figsize=default):
        # get necessary data
        labels = [
            "all trials",
            "block {}%".format(int(row["rw block 1"] * 100)),
            "block {}%".format(int(row["rw block 2"] * 100)),
            "block {}%".format(int(row["rw block 3"] * 100)),
        ]
        before = [
            row[f"{trials} before"],
            row[f"1 {trials} before"],
            row[f"2 {trials} before"],
            row[f"3 {trials} before"],
        ]
        across = [
            row[f"{trials} across"],
            row[f"1 {trials} across"],
            row[f"2 {trials} across"],
            row[f"3 {trials} across"],
        ]
        after = [
            row[f"{trials} after"],
            row[f"1 {trials} after"],
            row[f"2 {trials} after"],
            row[f"3 {trials} after"],
        ]
        combined = [
            row[f"{trials} all"],
            row[f"1 {trials} all"],
            row[f"2 {trials} all"],
            row[f"3 {trials} all"],
        ]
        # for ploting
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        if fig == None and ax == None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        # plot bar for all events
        rects1 = ax.bar(x - 3 * (width / 2), combined, width, label="combined")
        rects2 = ax.bar(x - (width / 2), before, width, label="before event")
        rects3 = ax.bar(x + (width / 2), across, width, label="across event")
        rects4 = ax.bar(x + 3 * (width / 2), after, width, label="after event")
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Cumulative count")
        if x_label:
            ax.set_xlabel(" ")
        if title != False:
            ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        # add labels
        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)
        self.autolabel(rects3, ax)
        self.autolabel(rects4, ax)

        y_lim = ax.get_ylim()
        ax.set_ylim(0,y_lim[1]+10)

        fig.tight_layout()

        return fig, ax

    def plt_neuron_fingerprint_summary_all(
        self, data_dict, info_df, bins, trials, title
    ):
        # create figure and axis

        ## CREATE GRID & AXIS =========================
        width = 11
        fig = plt.figure(figsize=(width, width * (3 / 5)))
        # create main grid
        gs = gridspec.GridSpec(
            ncols=1,
            nrows=2,
            # width_ratios=[1, 1],
            height_ratios=[2, 1],
            hspace=0.5,  # ,wspace=0.2
        )
        # create first row grid
        gs0 = gs[0].subgridspec(1, 2,)  # wspace=0.3)
        # add 0.0 axis
        ax1 = fig.add_subplot(gs0[0])
        # add 0.1 axis
        ax2 = fig.add_subplot(gs0[1])

        # create second row major gird
        gs1 = gs[1].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.1)
        # second row minor grid 5 = colorbar
        # ax6 = fig.add_subplot(gs1[1])
        # second row minor grid 0-4 = fingerprintplots
        # second row plots
        gs11 = gs1[0].subgridspec(1, 4, wspace=0.1)  # ,wspace = 0.1)
        # add 11.0-11.5 axis
        axs35 = list()
        for i in range(4):
            axs35.append(fig.add_subplot(gs11[0, i]))

        ims = list()

        ## PLOT TO AXS ================================
        # gernarate data
        # gen labels and ticks
        ticks = np.arange(0, bins + 1, 10)
        labels = ticks.astype(str).tolist()
        # data = data_dict[trials]["reward_alinged"]
        # lower = data_dict[trials]["percentiles"][0]
        # upper = data_dict[trials]["percentiles"][4]
        # fingerprint = self.get_fingerprint(data,lower,upper,bins)
        fingerprint = data_dict[trials]["fingerprint_per"]

        # axis 0.0 ========
        _, ax1 = self.plt_bar_sig_neurons(
            info_df.loc[1, :], trials, "Significant neurons", fig, ax1
        )

        # axis 0.2 ========
        _, ax2, im2 = self.plt_fingerprint_2d(
            fingerprint, ["95th percentile","5th percentile"], fig, ax2, "Outside of 90% of the data"
        )
        # set colorbar for axis 0.2
        fig.colorbar(im2, ax=ax2, orientation="vertical")

        # axis 1.0 ========
        _, axs35[0], im_ = self.plt_neuron_fingerprint_all(
            fingerprint.sort_values(by=["above"], ascending=False).iloc[:, 0:50].values,
            "all",
            fig,
            axs35[0],
        )
        # set xticks adn labels
        axs35[0].set_xticks(ticks)
        axs35[0].set_xticklabels(labels)
        ims.append(im_)

        # axis 1.1-1.4 ====
        for block, ax in zip([1, 2, 3], axs35[1:4]):
            # get data
            key = f"block{block}_{trials}"
            # data_ = data_dict[key]["reward_alinged"]
            # lower_ = data_dict[key]["percentiles"][0]
            # upper_ = data_dict[key]["percentiles"][4]
            # fingerprint_ = self.get_fingerprint(data_,lower_,upper_,bins)
            fingerprint_ = data_dict[key]["fingerprint_per"]
            data = (
                fingerprint_.sort_values(by=["above"], ascending=False)
                .iloc[:, 0:50]
                .values
            )
            # plot
            blocks = self.selected_trials_df["probability"].unique()
            _, ax, im = self.plt_neuron_fingerprint_all(
                data, f"block {int(blocks[block-1]*100)}%", fig, ax
            )
            # set xticks
            ax.set_xticks(ticks)
            # set xticklabels
            ax.set_xticklabels(labels)
            ims.append(im)

        # add colorbar to 1.2
        # clean ax6
        cb_ax = fig.add_axes([0.848, 0.11, 0.012, 0.205])
        cbar = fig.colorbar(
            ims[0], cax=cb_ax, ticks=np.arange(np.min(data), np.max(data) + 1)
        )
        cbar.ax.set_yticklabels(["below", "in", "above"])
        # add elgend to 1.4
        # axs35[3].legend()

        # clean up labels and ticks
        # remove from all
        for ax_ in axs35[1:]:
            plt.setp(ax_.get_yticklabels(), visible=False)
            plt.setp(ax_.yaxis.get_label(), visible=False)
            # plt.setp(ax_.xaxis.get_label(), visible=False)
        # remove x label from 1.0
        # plt.setp(axs35[0].xaxis.get_label(), visible=False)

        fig.suptitle(f"Neurons responding to {title}")

        axs = [ax1, ax2] + axs35

        return fig, axs

    def plt_neuron_fingerprint_summary_all_expanded(
        self, data_dict, info_df, bins, trials, title
    ):
        # create figure and axis
        # gs = 1col x 2row
        # ------------------------
        # gs0=gs[0]=2col x 1row
        # ax1 | ax2
        # -----------------------
        # gs1=gs[1]=2col x 1row
        # gs10=gs1[0]                | gs11=gs1[1]
        # gs100=gs10[0]=4col x 1 row |

        ## CREATE GRID & AXIS =========================
        width = 11
        fig = plt.figure(figsize=(width, width * (3 / 5)))

        # GRID 0# create main grid around all
        gs = gridspec.GridSpec(
            ncols=1,
            nrows=2,
            # width_ratios=[1, 1],
            height_ratios=[1, 1],
            hspace=0.5,  # ,wspace=0.2
        )
        # GRID 0 row 1#
        gs0 = gs[0].subgridspec(ncols=2, nrows=1,)  # wspace=0.3)
        # add 0.0 axis
        ax1 = fig.add_subplot(gs0[0])
        # add 0.1 axis
        ax2 = fig.add_subplot(gs0[1])

        # GRID 0 row 2#
        gs1 = gs[1].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.1)
        # GRID 1 row
        gs10 = gs1[0].subgridspec(
            nrows=2, ncols=1, wspace=0.1, hspace=0.2
        )  # ,wspace = 0.1)
        # GRID 2 -> 4 rows = 0per
        gs100 = gs10[0].subgridspec(nrows=1, ncols=4, wspace=0.1)  # ,wspace = 0.1)
        # GRID 3 -> 4 rows = sigma
        gs101 = gs10[1].subgridspec(nrows=1, ncols=4, wspace=0.1)  # ,wspace = 0.1)

        # add 11.0-11.5 axis
        axs35 = list()
        for i in range(4):
            axs35.append(fig.add_subplot(gs100[0, i]))
        # add 11.0-11.5 axis
        axs79 = list()
        for i in range(4):
            axs79.append(fig.add_subplot(gs101[0, i]))

        ims = list()

        ## PLOT TO AXS ================================
        # gernarate data
        # gen labels and ticks
        ticks = np.arange(0, bins + 1, 10)
        labels = ticks.astype(str).tolist()
        # data = data_dict[trials]["reward_alinged"]
        # lower = data_dict[trials]["percentiles"][0]
        # upper = data_dict[trials]["percentiles"][4]
        # fingerprint = self.get_fingerprint(data,lower,upper,bins)
        fingerprint = data_dict[trials]["fingerprint_per"]

        # 1 Row =====================================================================

        # axis 0.0 ========
        _, ax1 = self.plt_bar_sig_neurons(
            info_df.loc[1, :], trials, "Significant neurons", fig, ax1
        )

        # axis 0.2 ========
        _, ax2, im2 = self.plt_fingerprint_2d(
            fingerprint, ["95th percentile","5th percentile"], fig, ax2, "Outside of 90% of the data"
        )
        # set colorbar for axis 0.2
        fig.colorbar(im2, ax=ax2, orientation="vertical")

        # 2 Row =====================================================================

        # axis 1.0 =======
        _, axs35[0], im_ = self.plt_neuron_fingerprint_all(
            fingerprint.sort_values(by=["above"], ascending=False).iloc[:, 0:50].values,
            "all",
            fig,
            axs35[0],
            ylabel="5th-90th perc.",
        )
        # set xticks adn labels
        axs35[0].set_xticks(ticks)
        axs35[0].set_xticklabels(labels)
        plt.setp(axs35[0].get_xticklabels(), visible=False)
        plt.setp(axs35[0].xaxis.get_label(), visible=False)
        ims.append(im_)

        # axis 1.1-1.4 ====
        for block, ax in zip([1, 2, 3], axs35[1:4]):
            # get data
            key = f"block{block}_{trials}"
            # data_ = data_dict[key]["reward_alinged"]
            # lower_ = data_dict[key]["percentiles"][0]
            # upper_ = data_dict[key]["percentiles"][4]
            # fingerprint_ = self.get_fingerprint(data_,lower_,upper_,bins)
            fingerprint_ = data_dict[key]["fingerprint_per"]
            data = (
                fingerprint_.sort_values(by=["above"], ascending=False)
                .iloc[:, 0:50]
                .values
            )
            # plot
            blocks = self.selected_trials_df["probability"].unique()
            _, ax, im = self.plt_neuron_fingerprint_all(
                data, f"Block {blocks[block-1]}%", fig, ax, ylabel="5th-90th perc."
            )
            # set xticks
            ax.set_xticks(ticks)
            # set xticklabels
            ax.set_xticklabels(labels)
            ims.append(im)

        # clean up labels and ticks
        # remove from all
        for ax_ in axs35[1:]:
            plt.setp(ax_.get_yticklabels(), visible=False)
            plt.setp(ax_.yaxis.get_label(), visible=False)
            # plt.setp(ax_.xaxis.get_label(), visible=False)
            # plt.setp(ax_.get_xticklabels(), visible=False)
        # remove x label from 1.0
        # plt.setp(axs35[0].xaxis.get_label(), visible=False)

        # 3 Row =====================================================================
        fingerprint = data_dict[trials]["fingerprint_sig"]
        # axis 1.0 =======
        _, axs79[0], im_ = self.plt_neuron_fingerprint_all(
            fingerprint.sort_values(by=["above"], ascending=False).iloc[:, 0:50].values,
            False,
            fig,
            axs79[0],
            ylabel="2 sigma",
        )
        # set xticks adn labels
        axs79[0].set_xticks(ticks)
        axs79[0].set_xticklabels(labels)
        ims.append(im_)

        # axis 1.1-1.4 ====
        for block, ax in zip([1, 2, 3], axs79[1:4]):
            # get data
            key = f"block{block}_{trials}"
            # data_ = data_dict[key]["reward_alinged"]
            # lower_ = data_dict[key]["percentiles"][0]
            # upper_ = data_dict[key]["percentiles"][4]
            # fingerprint_ = self.get_fingerprint(data_,lower_,upper_,bins)
            fingerprint_ = data_dict[key]["fingerprint_sig"]
            data = (
                fingerprint_.sort_values(by=["above"], ascending=False)
                .iloc[:, 0:50]
                .values
            )
            # plot
            blocks = self.selected_trials_df["probability"].unique()
            _, ax, im = self.plt_neuron_fingerprint_all(
                data, False, fig, ax, ylabel="2 sigma"
            )
            # set xticks
            ax.set_xticks(ticks)
            # set xticklabels
            ax.set_xticklabels(labels)
            ims.append(im)

        # add colorbar to 1.2
        # clean ax6
        cb_ax2 = fig.add_axes([0.848, 0.11, 0.009, 0.31])
        cbar2 = fig.colorbar(
            ims[0], cax=cb_ax2, ticks=np.arange(np.min(data), np.max(data) + 1)
        )
        cbar2.ax.set_yticklabels(["below", "in", "above"])
        # add elgend to 1.4
        axs79[3].legend()

        # clean up labels and ticks
        # remove from all
        for ax_ in axs79[1:]:
            plt.setp(ax_.get_yticklabels(), visible=False)
            plt.setp(ax_.yaxis.get_label(), visible=False)
            plt.setp(ax_.xaxis.get_label(), visible=False)
        # remove x label from 1.0
        # plt.setp(axs35[0].xaxis.get_label(), visible=False)

        fig.suptitle(f"Neurons responding to {title}")

        axs = [ax1, ax2] + axs35 + axs79
        return fig, axs

    def plt_fingerprint_overfiew_trial_selection_individual(
        self, data_dict, info_df, bins, trials, title, conf_int="90percentil"
    ):

        if conf_int == "90percentil":
            # lower = data_dict[trials]["percentiles"][0]
            # upper = data_dict[trials]["percentiles"][4]
            fingerprint = data_dict[trials]["fingerprint_per"]
            title_add = "90per"
        elif conf_int == "2sigma":
            # var_ar = np.var(data_dict[trials]["binned"],axis=2)
            # mean_ar = np.mean(data_dict[trials]["binned"],axis=2)
            # lower = mean_ar-2*var_ar
            # upper = mean_ar+2*var_ar
            fingerprint = data_dict[trials]["fingerprint_sig"]
            title_add = "2sig"
        else:
            print('error')
            return

        # generate data
        # data = data_dict[trials]["reward_alinged"]
        # fingerprint = self.get_fingerprint(data,lower,upper,bins)

        # plot 2d fingerprint
        fig, ax, im = self.plt_fingerprint_2d(fingerprint, title_add)

        # plot bar chart fingerprint resp compare
        fig, ax = self.plt_bar_sig_neurons(
            info_df.loc[1, :], trials, title=False #f"Neurons responding to {title}"
        )
        print('yes')
        self.save_fig(f"neur_fingerprint_resp_{title_add}_{trials}", fig)

        # plot colormap all fingerprints of neuron
        fig, ax, im = self.plt_neuron_fingerprint_all(
            fingerprint.sort_values(by=["above"], ascending=False).iloc[:, 0:50].values,
            title=False, #f"Neurons responding to {title}\nall trials {title_add}"
        )
        self.save_fig(f"neur_fingerprint_{title_add}_{trials}", fig)

        # blocks  & all trials
        for block in [1, 2, 3]:
            key = f"block{block}_{trials}"
            # data = data_dict[key]["reward_alinged"]
            # lower = data_dict[key]["percentiles"][0]
            # upper = data_dict[key]["percentiles"][4]
            # fingerprint = self.get_fingerprint(data,lower,upper,bins)
            fingerprint = data_dict[key]["fingerprint_per"]
            # plot
            blocks = self.selected_trials_df["probability"].unique()
            fig, ax, im = self.plt_neuron_fingerprint_all(
                fingerprint.sort_values(by=["above"], ascending=False)
                .iloc[:, 0:50]
                .values,
                title=False, #f"Neurons responding to {title}\nBlock {blocks[block-1]}% trials (5%-90%)"
            )
            # safe
            self.save_fig(f"neur_fingerprint_{title_add}_{block}_{trials}", fig)


    # Plot Colormap =============================================================================================================================
    
    def plt_colormap(self,data_df,figsize=default):
        x_labels = ['gamble reward', 'safe reward','gamble no-reward',  'safe no-reward']
        y_labels = ['75%','25%','12.5%']


        fig, ax = plt.subplots(1,1,figsize=figsize)
        #im = ax.imshow(data,norm=LogNorm())

        my_cmap = copy.copy(mpl.cm.get_cmap('viridis')) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        im = ax.imshow(data_df, 
                norm=mpl.colors.LogNorm(), 
                interpolation='nearest', 
                cmap=my_cmap,
                rasterized=True
                )

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
                text = ax.text(j, i, data_df[i, j],
                            ha="center", va="center", color="w")
                text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                            path_effects.Normal()])

        #ax.set_title("Distribution of Event difference")
        #fig.colorbar(im)
        fig.tight_layout()
        ax.grid(False)
        return fig,ax


    # Plot and safe all figures =================================================================================================================
    def save_fig(self, name, fig, file='png'):
        folder = self.folder + "/figures/all_figures"
        if file=='png':
            try:
                fig.savefig(
                    folder + "/" + name + ".png", dpi=200, format="png", bbox_inches="tight"
                )
            except:
                fig[0].savefig(
                    folder + "/" + name + ".png", dpi=200, format="png", bbox_inches="tight"
                )
        elif file=='pdf':
            try:
                fig.savefig(
                    folder + "/" + name + ".pdf", format="pdf", bbox_inches="tight"
                )
            except:
                fig[0].savefig(
                    folder + "/" + name + ".pdf", format="pdf", bbox_inches="tight"
                )



    # Generate all plots =================================================================================================================
    def safe_all_clusters_compare_random_fixed(
        self, window, bins, reward_aligned_ar, mean_ar, percentil_ar, name, clusters
    ):
        for cluster in clusters:
            file_name = name + f"_{self.get_cluster_name_from_neuron_idx(cluster)}"
            self.save_fig(
                file_name,
                (
                    self.plt_compare_random_fixed(
                        cluster, window, bins, reward_aligned_ar, mean_ar, percentil_ar
                    )
                )[0],
            )

    def safe_all_clusters_compare_random_fixed_sigma(
        self, window, bins, reward_aligned_ar, mean_ar, var_ar, name, clusters
    ):
        for cluster in clusters:
            file_name = file_name = (
                name + f"_{self.get_cluster_name_from_neuron_idx(cluster)}"
            )
            self.save_fig(
                file_name,
                (
                    self.plt_compare_random_fixed_sigma(
                        cluster, window, bins, reward_aligned_ar, mean_ar, var_ar
                    )
                ),
            )

    def generate_plots(self,
                        window,
                        iterations,
                        bins,
                        sig_number=2,
                        individual=False,
                        reload_data_dict=True,
                        reload_spikes_ar=False,
                        ):
        if reload_data_dict:
            # prepare necessary data arrays
            self.load_data_dict(window, iterations, bins, reload_spikes_ar)
            self.load_info_df(sig_number=sig_number)

        # generate conf neuron plots ==================================
        # iterate over all sub dicts & safe all individual neurons - percentile
        for _, value in self.data_dict.items():
            if individual:
                clusters = np.arange(self.spikes_per_cluster_ar.shape[0])
                print("mean vs 90per -> all neurons")
            else:
                clusters = (
                    value["fingerprint_per"][
                        (
                            value["fingerprint_per"]
                            .loc[:, ["below", "above"]]
                            .sum(axis=1)
                            >= 2
                        )
                    ]
                ).index
                print(f"{_} mean vs 90per -> significant neurons")
            # save neurons
            self.safe_all_clusters_compare_random_fixed(
                window,
                bins,
                value["reward_alinged"],
                value["mean"],
                value["percentiles"],
                value["filename"] + "_90per",
                clusters,
            )

        # iterate over all sub dicts & safe all individual neurons - variance
        for _, value in self.data_dict.items():
            if individual:
                clusters = np.arange(self.spikes_per_cluster_ar.shape[0])
                print("mean vs 2sigma -> all neurons")
            else:
                clusters = (
                    value["fingerprint_sig"][
                        (
                            value["fingerprint_sig"]
                            .loc[:, ["below", "above"]]
                            .sum(axis=1)
                            >= 2
                        )
                    ]
                ).index
                print(f"{_} mean vs 2sigma -> significant neurons")
            self.safe_all_clusters_compare_random_fixed_sigma(
                window,
                bins,
                value["reward_alinged"],
                np.mean(value["binned"], axis=2),
                np.var(value["binned"], axis=2),
                value["filename"] + "_2sigma",
                clusters,
            )

        # gernerate neuron fingerprint plots ============================
        print(f"{_} fingerprint summary & individual -> all subselections")
        for trials, name in zip(
            [
                "all",
                "rw",
                "norw",
                "gamble",
                "rw_gamble",
                "norw_gamble",
                "safe",
                "rw_safe",
                "norw_safe",
            ],
            [
                "reward and no-reward events",
                "reward events",
                "no-reward events",
                "gamble side and reword and no-reward events",
                "gamble side and reward events",
                "gamble side and no-reward events",
                "safe side and reword and no-reward events",
                "safe side and reward events",
                "safe side and no-reward events",
            ],
        ):
            # gemerate summary
            fig, _ = self.plt_neuron_fingerprint_summary_all(
                self.data_dict, self.info_df, bins, trials, name
            )
            self.save_fig(f"fingerprint_summary_{trials}", fig)
            # generate individual
            self.plt_fingerprint_overfiew_trial_selection_individual(
                self.data_dict, self.info_df, bins, trials, name, conf_int="90percentil"
            )
            self.plt_fingerprint_overfiew_trial_selection_individual(
                self.data_dict, self.info_df, bins, trials, name, conf_int="2sigma"
            )

    def generate_fingerprint_plots(self,
                        window,
                        iterations,
                        bins,
                        sig_number=2,
                        reload_data_dict=True,
                        reload_spikes_ar=False,
                        ):
        if reload_data_dict:
            # prepare necessary data arrays
            self.load_data_dict(window, iterations, bins, reload_spikes_ar)
            self.load_info_df(sig_number=sig_number)
            self.info_df.loc[:,['all all']]=51
            self.info_df.loc[:,['1 all all']]=19

        for trials, name in zip(
            [
                "all",
                "rw",
                "norw",
                "gamble",
                "rw_gamble",
                "norw_gamble",
                "safe",
                "rw_safe",
                "norw_safe",
            ],
            [
                "reward and no-reward events",
                "reward events",
                "no-reward events",
                "gamble side and reword and no-reward events",
                "gamble side and reward events",
                "gamble side and no-reward events",
                "safe side and reword and no-reward events",
                "safe side and reward events",
                "safe side and no-reward events",
            ],
        ):
            # gemerate summary
            fig, _ = self.plt_neuron_fingerprint_summary_all(
                self.data_dict, self.info_df, bins, trials, name
            )
            self.save_fig(f"fingerprint_summary_{trials}", fig)
            # generate individual
            self.plt_fingerprint_overfiew_trial_selection_individual(
                self.data_dict, self.info_df, bins, trials, name, conf_int="90percentil"
            )
            self.plt_fingerprint_overfiew_trial_selection_individual(
                self.data_dict, self.info_df, bins, trials, name, conf_int="2sigma"
            )


