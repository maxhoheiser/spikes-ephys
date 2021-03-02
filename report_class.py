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


# class ###################################################################################################################
class SpikesReport():
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

# Save all & Create Report ===================================================================================================
   

    #============================================================================================================================
    # generate latex report

    # create latex report with all images
    def image_box_cluster(self, file_name, cluster, width=0.4, last=False):
        arg = (r"\parbox[c]{1em}{\includegraphics[width="+ str(width)+r"\textwidth]{"+self.folder+r"/figures/all_figures/"+file_name+r"_"+str(cluster)+r".png}}")
        if last:
            arg += r"\\"
        else:
            arg += "&"
        return arg
    
    def image_box(self, file_name, width=0.4, last=False):
        arg = (r"\parbox[c]{1em}{\includegraphics[width="+ str(width)+r"\textwidth]{"+self.folder+r"/figures/all_figures/"+file_name+r".png}}")
        if last:
            arg += r"\\"
        else:
            arg += "&"
        return arg
        return arg

    def generate_report(self):
        # prepareation variables
        block = self.selected_trials_df['probability'].unique()
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

        # summary section
        with doc.create(Section('Session summary')):
            # create summary table
            with doc.create(LongTabu("X | X")) as summary_table:
                with doc.create(Tabular("r r")) as small_table:
                    small_table.add_row(["Summary",""])
                    small_table.add_hline()
                    small_table.add_row(["Gamble side:", self.gamble_side])
                    small_table.add_hline()
                    small_table.add_row(["All trials", self.all_trials_df.index.max()])
                    small_table.add_row(["Good trials", self.good_trials_df.index.max()])
                    small_table.add_row(["Selected trials", self.selected_trials_df.index.max()])
                    small_table.add_hline()
                    small_table.add_row(["Probability bins", str(self.all_trials_df['probability'].unique())])
            
                # add overview plots
                doc.append(NoEscape( self.image_box("hist_fit", last=True) ))
                doc.append(NoEscape( self.image_box("trial_length")) )
                doc.append(NoEscape( self.image_box("cluster_hist", last=True) ))
                doc.append(NewPage())


        # Add stuff to the document
        with doc.create(Section('Spike Trains and Histogram for Reward Events')):
                # create necessary variables
                for cluster in self.clusters_df.loc[self.clusters_df['group']=='good'].index:
                    # create subsection title
                    subsection = "Cluster " + str(cluster)
                    with doc.create(Subsection(subsection, label=False)):
                        # create details table
                        with doc.create(LongTabu("X | X")) as details_table:
                            doc.append(NoEscape( self.image_box_cluster("isi",cluster) )) 
                            doc.append(NoEscape( self.image_box_cluster("spk_train",cluster, last=True) )) 
                            details_table.add_hline()
                            details_table.add_row(["All Trials", "Rewarded Trials"])
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_all-events",cluster) )) 
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_all-events_reward-centered",cluster, last=True) )) 
                            details_table.add_hline()
                            details_table.add_row(["Gambl Side Reward", "Save Side Reward"])
                            #details_table.end_table_header()
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_gamble_reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_save_reward", cluster, last=True) ))
                            #details_table.add_hline()
                            details_table.add_row(["Gambl Side No-Reward", "Save Side No-Reward"])
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_gamble_no-reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("spk_train_hist_save_no-reward",cluster, last=True) ))
                        doc.append(NewPage())

                        with doc.create(LongTabu("X | X")) as bootstrap_table:
                            doc.append('Reward aligned compard to random aligned\newline\newline')
                            bootstrap_table.add_row(["Gamble-Side", "Save-Side"])
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_gamble",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_save", cluster, last=True) ))
                            bootstrap_table.add_hline()
                            bootstrap_table.add_row(["Reward", "No-Reward"])
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_no_reward", cluster, last=True) ))
                            # block 1
                            bootstrap_table.add_hline()
                            bootstrap_table.add_row([f"Block 1: {block[0]}%",""])
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block1_reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block1_no_reward", cluster, last=True) ))
                            doc.append(NewPage())
                            # block 2
                            bootstrap_table.add_hline()
                            bootstrap_table.add_row([f"Block 2: {block[1]}%",""])
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block2_reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block2_no_reward", cluster, last=True) ))
                            # block 3
                            bootstrap_table.add_hline()
                            bootstrap_table.add_row([f"Block 3: {block[2]}%",""])
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block3_reward",cluster) ))
                            doc.append(NoEscape( self.image_box_cluster("reward_aligned_block3_no_reward", cluster, last=True) ))
                    doc.append(NewPage())

        # create file_name
        filepath = (self.folder+"/"+self.session+"-report")
        # create pdf
        doc.generate_pdf(filepath, clean=True, clean_tex=True)#, compiler='latexmk -f -xelatex -interaction=nonstopmode')
        #doc.generate_tex(filepath)

    # create interactive webpage







# ODL

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

    