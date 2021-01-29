import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import importlib
import os
import sys
import platform
import datetime

class SyncPhenosys():
    """[# synchronisation class for Phenosys Behavior Recording and Neuron Electrophysiology Recording]
    """    
    def __init__(self, session, folder, channel_no=6, info_channel=1, rows_missing_ttl=[]):
        """[summary]

        Args:
            session ([type]): [description]
            folder ([type]): [description]
            channel_no (int, optional): [description]. Defaults to 6.
            info_channel (int, optional): [description]in. Defaults to 1.
        """        
        self.session = session
        self.folder = folder
        self.channel_no = channel_no
        self.ttl_channels = self.load_digitalin()
        self.ttl_signals = self.ttl_create_ticks()
        self.ttl_event_dict=self.create_dict()
        self.ttl_info_channel = self.convert_ttl_to_event('channel '+str(info_channel))
        self.csv = self.load_csv()
        self.rows_missing_ttl = rows_missing_ttl
        self.combined_df = self.combine_dataframes()
        self.all_trials_df, self.good_trials_df =  self.get_trials()


 # Load & manipulate Intern binary Data ====================================================================
    # load neuron binary files to array
    def load_digitalin(self):
        with open(self.folder+'/electrophysiology/digitalin.dat', 'r') as f:
            #a = np.fromfile(f, dtype=np.uint32)
            binary = np.fromfile(f, dtype=np.uint16)
        # get channels from 
        ttl_channels=pd.DataFrame()
        def get_channel(array, n):
            return (array & (1<<n))>>n
        for channel in range(self.channel_no):
            ttl_channels['channel '+str(channel)]=get_channel(binary, channel)
        ttl_channels.index.name = 'Sampling rate 20kHz'
        return ttl_channels

    # find length of ttl signal
    def ttl_find_lenght(self, data_frame, column, zeros=False):
        # calculate length of ttl signlas for each frame
        df = data_frame
        frame = column
        change = np.where(df[frame].values[:-1] != df[frame].values[1:])[0]+1
        change = np.insert(change, 0, 0)
        values = df.loc[change, frame]
        diff = np.diff(change)
        last = df.shape[0] - change[-1]
        diff= np.append(diff, last)
        output_df = pd.DataFrame({'Start':change, 'Value':values, 'Length':diff})
        output_df.reset_index(inplace=True, drop=True)
        if zeros:
            return output_df
        else:
            output_df = output_df.loc[output_df['Value']>0,:]
            output_df.drop('Value', axis=1, inplace=True)
            return output_df

    # create data frame with ttl ticks for each channels
    def ttl_create_ticks(self):
        ttl_signals = dict()
        for key in self.ttl_channels.columns:
            data = self.ttl_find_lenght(self.ttl_channels, key)
            ttl_signals[key]=data
        return ttl_signals


    # convert ttil to events ======================
    # event & time dict
    def create_dict(self):
        durr_range = dict()

        # old trial dict
        # durr_range['TIstarts']=(11,29)
        # durr_range['IND-CUE_pres_start']=(31,49)
        # durr_range['SOUND_start']=(51,69)
        # durr_range['resp-time-window_start']=(71, 89)
        # durr_range['right_rewarded']=(91,110)
        # durr_range['right_NOreward']=(111,129)
        # durr_range['left_rewarded']=(131,149)
        # durr_range['left_NOreward']=(151,169)
        # durr_range['no response in time']=(173,186)
        # durr_range['ITIstarts']=(190,213)
        # durr_range['ITIends']=(215,245)
        
        durr_range['start']=(11,29)
        durr_range['cue']=(31,49)
        durr_range['sound']=(51,69)
        durr_range['openloop']=(71, 89)
        durr_range['right_rw']=(91,110)
        durr_range['right_norw']=(111,129)
        durr_range['left_rw']=(131,149)
        durr_range['left_norw']=(151,169)
        durr_range['no response in time']=(173,186)
        durr_range['iti']=(190,213)
        durr_range['end']=(215,245)
        return durr_range

    # helper function to convert each value to event
    def convert_durration_to_event(self, durr):
        for key, (start,stop) in self.ttl_event_dict.items():
            if durr>=start and durr<=stop:
                return key

    # convert ttl length to events
    def convert_ttl_to_event(self, channel):
        self.ttl_signals[channel]['Event'] = self.ttl_signals[channel]['Length'].apply(self.convert_durration_to_event)
        return self.ttl_signals[channel]


 # Load & manipulate Neuron binary Data ====================================================================
    # convert to datetime format with ms
    def convert_to_datetime(self, excel_string):
        second = (excel_string-25569)*86400.0
        return datetime.datetime.utcfromtimestamp(second)

    # find probability function
    def match_probability(self, df, start, stop):
        if "prob75" in (df.loc[stop]['Probability']):
            df.loc[ start:stop, 'Probability' ] =0.75
        elif "prob25" in (df.loc[stop]['Probability']):
            df.loc[ start:stop, 'Probability' ] =0.25
        elif "prob12" in (df.loc[stop]['Probability']):
            df.loc[ start:stop, 'Probability' ] =0.125

    #load csv file======================
    def load_csv(self):
        csv_file = self.folder+'/behavior/output.csv'
        csv = pd.read_csv(csv_file, delimiter=',', encoding='utf-16', header=0, skiprows=[1])
        csv.columns=['Event Time', 'Event', 'Probability', 'Side']
        
        # get gamble side
        gamble_string = csv.loc[ csv['Side'].notnull(), 'Side'].values[0]
        if 'RIGHT' in gamble_string:
            self.gamble_side = 'right'
        if 'LEFT' in gamble_string:
            self.gamble_side = 'left'
    
        # drop side column
        csv.drop('Side', axis=1, inplace=True)

        # Cleanup DateTime
        csv['Event Time'] = csv['Event Time'].apply(self.convert_to_datetime)
        start_dateteime = csv.loc[0, 'Event Time']

        # convert ms to sampling rate time delta
        delta = csv['Event Time'] - csv.loc[0, 'Event Time']
        csv.insert (1, 'Start', (delta.dt.total_seconds()*20000).astype('uint64') )

        # clean up proabability column =====
        # calculate where prob changes
        prob = csv.loc[csv['Probability'].notnull(),'Probability']
        prob_change = np.where(prob.values[:-1] != prob.values[1:])[0]
        prob_change_idx = prob.iloc[prob_change].index.values
        prob_change_idx = np.append(prob_change_idx, prob.index[-1])
        # change 3 bins probability to number
        # change first bin
        start = 0
        stop = prob_change_idx[0]
        self.match_probability(csv, start, stop)
        # change second bin
        start = prob_change_idx[0]+1
        stop = prob_change_idx[1]
        self.match_probability(csv, start, stop)
        # change third bin
        start = prob_change_idx[1]+1
        stop = stop = prob_change_idx[2]
        self.match_probability(csv, start, stop)
        # add probability to last rows
        nan = np.where(csv['Probability'].isnull())[0]
        csv.loc[nan[0]:, 'Probability'] = csv.loc[nan[0]-1, 'Probability']

        # cleanup event names
        # new names dict
        replace = dict()
        replace['TIstarts']='start'
        replace['IND-CUE_pres_start']='cue'
        replace['SOUND_start']='sound'
        replace['resp-time-window_start']='openloop'
        replace['right_rewarded']='right_rw'
        replace['right_NOreward']='right_norw'
        replace['left_rewarded']='left_rw'
        replace['left_NOreward']='left_norw'
        replace['no response in time']='no response in time'
        replace['ITIstarts']='iti'
        replace['ITIends']='end'
        replace['start'] = 'session start'
        replace['end'] = 'session end'
        csv['Event'] = csv['Event'].apply(lambda event: replace[event] if event in replace.keys() else event)

        return csv


 # Align and Find Symmetry =================================================================================
    # helper function to insert a nan value to rows missing
    def Insert_row(self, row_number, df, row_value, column='all'): 
        # Starting value of upper half 
        start_upper = 0
        # End value of upper half 
        end_upper = row_number 
        # Start value of lower half 
        start_lower = row_number 
        # End value of lower half 
        end_lower = df.shape[0] 
        # Create a list of upper_half index 
        upper_half = [*range(start_upper, end_upper, 1)] 
        # Create a list of lower_half index 
        lower_half = [*range(start_lower, end_lower, 1)] 
        # Increment the value of lower half by 1 
        lower_half = [x.__add__(1) for x in lower_half] 
        # Combine the two lists 
        index_ = upper_half + lower_half 
        # Update the index of the dataframe 
        df.index = index_ 
        # Insert a row at the end 
        df.loc[row_number] = row_value 
        # Sort the index labels 
        df = df.sort_index() 
        # return the dataframe 
        return df 

    # create combined dataframe
    def combine_dataframes(self, align=False):

        ttl_combined = self.ttl_signals['channel 1'].copy()
        ttl_combined.columns=(['TTL Start', 'TTL Length', 'TTL Event'])

        for row in self.rows_missing_ttl:
            ttl_combined = self.insert_row(row, ttl_combined, np.nan, column='all')

        ttl_combined.reset_index(inplace=True, drop=True)
        ttl_combined['TTL Start norm'] = ttl_combined['TTL Start']-ttl_combined.loc[0, 'TTL Start']
        ttl_combined['TTL index']=ttl_combined.index

        not_in_ttl = self.csv['Event'].unique()[~np.isin(self.csv['Event'].unique(), self.ttl_signals['channel 1']['Event'].unique())]
        csv_combined = self.csv.loc[ (self.csv['Event']!=not_in_ttl[0]) & (self.csv['Event']!=not_in_ttl[1]) & (self.csv['Event']!=not_in_ttl[2]) ].copy()
        csv_combined.drop('Event Time', axis=1, inplace=True)
        csv_combined.columns=(['CSV Start', 'CSV Event', 'CSV Probability'])
        csv_combined.reset_index(inplace=True, drop=True)
        csv_combined['CSV Start norm'] = csv_combined['CSV Start']-csv_combined.loc[0, 'CSV Start']
        csv_combined['CSV index']=csv_combined.index

        combined = pd.merge(ttl_combined, csv_combined, how='outer', left_index=True, right_index=True)
        combined['Delta (TTL-CSV)'] = combined['TTL Start norm']-combined['CSV Start norm']
        combined['Compare'] = combined['TTL Event']==combined['CSV Event']

        if align:
            return combined[['TTL Event','CSV Event','TTL Start norm','CSV Start norm','Delta (TTL-CSV)','TTL index','CSV index']]
        
        else:
            #csv = self.load_csv()
            #print(csv.loc[(csv.loc[:,'Event']=='wheel is not stopping'),  'Start'].values)
            # add "wheel not stopping" event for each start row
            n_stp_time = self.csv.loc[(self.csv.loc[:,'Event']=='wheel is not stopping'),  'Start'].values
            i = 0
            j = 0
            while i < combined.shape[0]-1:
                if combined.loc[i,'CSV Event']=='start' and  combined.loc[i+1,'CSV Event']=='start':
                    combined = self.insert_row(i+1, 
                                            combined, 
                                            [n_stp_time[j],'wheel not stopping'],
                                            ['CSV Start', 'CSV Event']
                                            )
                    # att wheel not stopping csv index
                    j += 1
                # add running index
                i += 1
            
            combined['CSV Start norm'] = combined['CSV Start']-combined.loc[0, 'CSV Start']
            
            
            # do not uncomment
            """ 
            # calculate trial number and set index acording to number
            combined['index']=combined.index
            combined['Trial']=np.nan
            trial=1
            for index, row in combined.iterrows():
                l1 = ['start', 'cue', 'sound', 'openloop']
                l2 = ['iti', 'end']
                if (row['CSV Event'] == 'start') & ( list(combined.loc[index:(index+3)]['CSV Event'].values)==l1 )  & ( list(combined.loc[(index+5):(index+6)]['CSV Event'].values)==l2 ):
                    combined.loc[index:(index+6),'Trial']=trial
                    trial+=1
                else:
                    row['Trial']=np.nan
            combined.set_index(['Trial', 'index'], inplace=True)

            # calculate index of event in each trial and set index
            combined.set_index((combined.groupby(level=0).cumcount()).rename('Group Index'), append=True, inplace=True) """

            # calculate trial number and set index acording to number
            combined['index']=combined.index
            combined['Trial']=np.nan
            # add good or bad trial
            all_trial = 0
            trial = 0
            for index, row in combined.iterrows():
                l1 = ['start', 'cue', 'sound', 'openloop']
                l2 = ['iti', 'end']
                l3 = ['start', 'wheel not stopping']
                if (row['CSV Event'] == 'start') & ( list(combined.loc[index:(index+3)]['CSV Event'].values)==l1 )  & ( list(combined.loc[(index+5):(index+6)]['CSV Event'].values)==l2 ):
                    combined.loc[index:(index+6),'Good Trial']=True
                    combined.loc[index:(index+6),'Trial']=trial
                    combined.loc[index:(index+6),'All Trial']=all_trial
                    trial+=1
                    all_trial+=1
                elif (row['CSV Event'] == 'start') & (list(combined.loc[index:(index+1)]['CSV Event'].values)==l3):
                    combined.loc[index:(index+1),'Good Trial']=False
                    combined.loc[index:(index+1),'Trial']=np.nan
                    combined.loc[index:(index+1),'All Trial']=all_trial
                    all_trial+=1
                else:
                    row['Good Trial']=False
                    row['All Trial']=np.nan


            combined.set_index(['All Trial', 'Trial', 'index'], inplace=True)

            # calculate index of event in each trial and set index
            combined.set_index((combined.groupby(level=0).cumcount()).rename('Group Index'), append=True, inplace=True)

            return combined


    # Function to insert row in the dataframe 
    def insert_row(self, row_number, df, row_value, column='all'): 
        # Starting value of upper half 
        start_upper = 0
        # End value of upper half 
        end_upper = row_number 
        # Start value of lower half 
        start_lower = row_number 
        # End value of lower half 
        end_lower = df.shape[0] 
        # Create a list of upper_half index 
        upper_half = [*range(start_upper, end_upper, 1)] 
        # Create a list of lower_half index 
        lower_half = [*range(start_lower, end_lower, 1)] 
        # Increment the value of lower half by 1 
        lower_half = [x.__add__(1) for x in lower_half] 
        # Combine the two lists 
        index_ = upper_half + lower_half 
        # Update the index of the dataframe 
        df.index = index_ 
        # Insert a row at the end 
        if column == 'all':
            df.loc[row_number,:] = row_value  
        else:
            df.loc[row_number,column] = row_value  
        # Sort the index labels 
        df = df.sort_index() 
        df.reset_index()
        # return the dataframe 
        return df 

    # get good trials
    """def get_trials(self, combined):
        trials = combined.loc[~np.isnan(combined.index.get_level_values('Trial')),['TTL Start', 'CSV Event']]
        trials.columns = ['Start', 'Event']

        return trials"""

    # get trials
    # convert combined to trials including wheel not stopping
    def get_trials(self,incl_wheel_ns=True):
        #fix combined
        ttl_norm = self.combined_df.loc[pd.IndexSlice[0,:,:,0],'TTL Start'].values[0]-self.combined_df.loc[pd.IndexSlice[0,:,:,0],'CSV Start'].values[0]
        current_delta = 0
        for index, row in self.combined_df.iterrows():
            # patch ttl missing values
            if np.isnan(row['TTL Start']):
                self.combined_df.loc[index,'TTL Start'] = row['CSV Start']+ttl_norm+current_delta
            else:
                current_delta = row['Delta (TTL-CSV)']


        trials_df = pd.DataFrame(columns=['index_all_trials','index_good_trials','start', 'cue', 'sound', 'openloop', 'reward', 'iti', 'end', 'event',
                                'probability', 'length', 'select'])
        trials_df['select']=True
        
        # iterate overall grouped frames
        for group, frame in self.combined_df.groupby(level=0):
            ttl_start = frame['TTL Start'].values
            times = list(map(int, ttl_start))
            length = int(ttl_start[-1]-ttl_start[0])
            index_all = int(frame.index[0][0])
            index_good = frame.index[0][1]
            
            if not np.isnan(index_good):
                index_good = int(index_good)

            if ttl_start.shape[0]==7:
                event  = frame.loc[pd.IndexSlice[:,:,:,4],'CSV Event'].values[0]
                #times = [int(i) if not np.isnan(i) else i for i in ttl_start ] 
                
                
            elif ttl_start.shape[0]==2:
                event = 'wheel not stopping'
                times = times + [np.nan, np.nan, np.nan, np.nan, np.nan]
                #csv_start = frame.loc[pd.IndexSlice[:,:,:,1],'CSV Start'].values[0]
                #delta = frame.loc[pd.IndexSlice[:,:,:,0],'Delta (TTL-CSV)'].values[0]
                #length = int(csv_start + delta)

            probability = frame.loc[pd.IndexSlice[:,:,:,0],'CSV Probability'].values[0]

            new_row = [index_all, index_good] + times + [event, probability, length, True]

            trials_df.loc[trials_df.shape[0] + 1] = new_row

        # convert 20khz sampling point length to ms length
        trials_df['length_ms']=trials_df['length']*0.05

        # set index_all_trials as dataframe index
        trials_df.set_index('index_all_trials', inplace=True)

        # right left and reward big column
        trials_df['right']=False
        trials_df['left']=False
        trials_df['reward_given']=False

        trials_df
        trials_df.loc[trials_df['event']=='right_rw',['right','reward_given']]=True
        trials_df.loc[trials_df['event']=='right_norw','right']=True
        trials_df.loc[trials_df['event']=='left_rw',['left','reward_given']]=True
        trials_df.loc[trials_df['event']=='left_norw','left']=True

        trials_df['good']=False
        trials_df.loc[trials_df['index_good_trials'].notna(),'good']=True

        # change to numeric
        trials_df[['start','cue','sound','openloop','reward','iti','end','probability','length']] = trials_df[['start','cue','sound','openloop','reward','iti','end','probability','length']].apply(pd.to_numeric)

        # create good trials dataframe
        good_trials_df = trials_df.loc[trials_df['good'],:]
        good_trials_df.set_index('index_good_trials',inplace=True)



        return (trials_df, good_trials_df)

    # get all trials including wheel not stopping and 







