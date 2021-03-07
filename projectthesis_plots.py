"""
Generate all plots for Project Thesis
"""

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

import csv
import scipy.stats as st
import importlib
import os
from os import path
import sys
import platform
import qgrid
import datetime
from scipy.interpolate import make_interp_spline, BSpline
import pickle
import random

import dill as pickle

from scipy.stats import chisquare
from numba import njit

from sync_class import SyncPhenosys
from sync_class import SyncPybpod
from eda_class import SpikesEDA
from behavior_class import BehaviorAnalysis
from sda_class import SpikesSDA
from report_class import SpikesReport

from analyze_phenosys import *

default = [6.4, 4.8]


windows_folder = r"C:/Users/User/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"
linux_folder = "/home/max/ExpanDrive/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"
mac_folder = "/Users/max/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/1 Data Analysis"

project_folder = "/Users/max/Google Drive/3 Projekte/Masterarbeit Laborarbeit Neuroscience/Schreiben/Projektarbeit/projektarbeit_latex"

textwidth = 418.25368
# Helper Functions=================================================================================

def save_fig(name, fig):
    folder = path.join(project_folder,'graphics')
    try:
        fig.savefig(
                    folder + "/" + name + ".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    pad_inches = 0,
                    )  
    except:
        fig[0].savefig(
                        folder + "/" + name + ".pdf",
                        format='pdf',
                        bbox_inches='tight'
                        ) 


def set_size(fraction=1,size='default'):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    default_ratio = 4.8/6.4
    # Figure width in inche
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if size=='quad':
        fig_height_in = fig_width_in
    elif size=='long':
        fig_height_in = fig_width_in * 0.4
    elif size=='xlong':
        fig_height_in = fig_width_in * 0.3
    else:
        fig_height_in = fig_width_in * 0.6875 #golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return (fig_width_in, fig_height_in)



# Load Sessions ====================================================================================================================
class  LoaderClass():
    """
    Class for loading multiple sessions to data dictionary
    each session = object with subobjects
    """
    def __init__(self,sync_obj,behavior_obj,eda_obj,sda_obj,report_obj):
        self.sync=sync_obj
        self.behavior=behavior_obj
        self.eda=eda_obj
        self.sda=sda_obj
        self.report=report_obj


def load_session(session, missing_rows_ttl=[], lo_spikes=False, deselect_trials=[]):
    """
    Load a session of data analysis
    Args:
        session (string): name of the session == folder
        missing_rows_ttl (list, optional): . Defaults to [].
        lo_spikes (bool, optional): . Defaults to False.
        deselect_trials (list, optional):  Defaults to [].
    Returns:
        object: object of type LoaderClass with all specified subobjects
    """
    # load calss and set folder depending on platform
    folder = get_session_folder(session)
    sync_obj = SyncPhenosys(session, folder, 7, 1, missing_rows_ttl) 
    behavior_obj = BehaviorAnalysis(sync_obj, deselect_trials)
    if lo_spikes:
        print(f"{session} -> sda")
        eda_obj = SpikesEDA(behavior_obj)
        sda_obj = SpikesSDA(eda_obj)
        report_obj = SpikesReport(sda_obj)
        session_obj = LoaderClass(sync_obj,behavior_obj,eda_obj,sda_obj,report_obj)
    else:
        print(f"{session} -> behavior")
        session_obj = LoaderClass(sync_obj,behavior_obj,None,None,None)
    return session_obj


def load_pb_session(session, pb_root, oe_root,lo_spikes=False):
    """
    Load a session of data analysis
    Args:
        session (string): name of the session == folder
        pb_root (path): . 
        oe_root (path): . 
    Returns:
        object: object of type LoaderClass with all specified subobjects
    """
    # load calss and set folder depending on platform
    folder = get_session_folder(session)
    sync_obj = SyncPybpod(session, pb_root, oe_root) 
    if lo_spikes:
        print(f"{session} -> sda")
        #eda_obj = SpikesEDA(behavior_obj)
        #sda_obj = SpikesSDA(eda_obj)
        #report_obj = SpikesReport(sda_obj)
        #session_obj = LoaderClass(sync_obj,behavior_obj,eda_obj,sda_obj,report_obj)
    else:
        print(f"{session} -> behavior")
        session_obj = LoaderClass(sync_obj,None,None,None,None)
    return session_obj


def get_session_folder(session):
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder + '/' + session
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder + r"/" + session
    elif platform.system() == 'Darwin':
        folder = mac_folder + r"/" + session     
    return folder

def get_folder():
    if platform.system() == 'Linux':
        # Linux
        folder = linux_folder
    elif platform.system() == 'Windows':
        # windows
        folder = windows_folder
    elif platform.system() == 'Darwin':
        folder = mac_folder    
    return folder