%% load spike times & clusters & manual clusters
% open all clusters to see what clusters there are
folder = 'C:\Users\User\Google Drive\3 Projekte\Masterarbeit Laborarbeit Neuroscience\1 Data Analysis\JG14_190621\electrophysiology';
spike_times = readNPY(fullfile(folder, 'spike_times.npy'));
spike_clusters = readNPY(fullfile(folder, 'spike_clusters.npy'));
cluster_group = tdfread(fullfile(folder, 'cluster_group.tsv'));
cluster_info = tdfread(fullfile(folder, 'cluster_info.tsv'));

% load ttl times and convert to necessary format
load('C:\Users\User\Google Drive\3 Projekte\Masterarbeit Laborarbeit Neuroscience\1 Data Analysis\JG14_190621\ttl_behavior_trials');
%ttl_times = ground_truth_trial;
load('C:\Users\User\Google Drive\3 Projekte\Masterarbeit Laborarbeit Neuroscience\1 Data Analysis\JG14_190621\ttl_times.mat');
clear('ground_truth_trial');

behavior_vec = ttl_times(:,[2,15]);
trial_info=trialsInfo(behavior_vec, folder);

%% shortcut to load workspace
load('C:\Users\Nutzer\Google Drive\1 Uni\1.3 Uni Projekte\Masterarbeit Laborarbeit Neuroscience\Data Analysis\JG14_190621\spikeanalysis\matlab.mat');
[trial_info.working_trials, trial_info.working_trials_bool] = workingTrialsGet(trial_info.trials, 10E4);

%% plot trial length & mean & std
% plot all trials
figure
histogram(trial_info.trials(:,3),50)

figure
plot(trial_info.trials(:,1), trial_info.trials(:,3))

%plot only trials within +- 3x std
%std_range = 2;
%working_trials = trial_info.trials( ( (abs(trial_info.trials(:,3))-abs(trial_info.stats.mean_trial_length)) <= std_range*trial_info.stats.std_trial_length ), :);

%trial_info.working_trials=workingTrialsGet(trial_info.trials, 10E4);
trial_info.working_trials_bool = zeros(size(trial_info.trials,1),1);
trial_info.working_trials=trial_info.trials(2:222,:);
trial_info.working_trials_bool(2:222,1)=1;
%trial_info.working_trials_bool([1,223:end],1)=0;




gambl_change  = gamblProbChange(folder, trial_info, working_trials);

figure
histogram(working_trials(:,3),50);


% inspect 1 abnormal trial
% x = 5

% trial_abnomral_indiv = ttl_times(find(ttl_times(:,12) == trial_abnormal(x,1)),:);
% trial_abnormal_ttl_around = find(abs(ttl_event_1_1(:,1)-trial_abnormal(x,3))< 100);
% trial_abnormal_ttl_around = ttl_event_1_1(find(abs(ttl_event_1_1(:,1)-trial_abnormal(x,3))< 500000),:);

%% plot spike train for 10 trial + trial begin and end mark
% test with custer 
% work with working_trials (+- 3sigma)

clusters = cluster_group.cluster_id( cluster_group.group(:) == 'g' );

for cluster = clusters'
    drawSpikeTrains(behavior_vec,cluster,folder,trial_info,0,0.6)
end
%
%

%% plot frequency for each trial
% frequency in trial_info.spikes.count0-xxx(:,3)

% #### ATENTION !!!! only plot spikes from working trials not all spikes !!

for cluster = cluster_group.cluster_id( cluster_group.group(:) == 'g' )'
    plotSpikeFrequency(cluster, trial_info, 0, folder)
end



%
%

%% plot density map 2d plot of frequency changes over all trials all

clusters = cluster_group.cluster_id( cluster_group.group(:) == 'g' );
%clusters(clusters(:,1)==17,:)=[];
x = size(clusters);

contourFrequencyPlot(clusters, trial_info, folder, x, 40)



%% Workspace ##############################################################

hold all
for i = 1:50
    plot([spike_vec(i,1) spike_vec(i,1)], [-0.4 0.4],'-b', 'LineWidth', 2)
end


f





x = unique(spike_clusters(:));
y = unique(cluster_group.cluster_id);
for i = 1:size(y)
    z(i,2)=y(i,1);
end
if size(x)~=size(y)
    print('Error in cluster size')
    return
end










% inspect and plot frequency per trial

% call for cluster and convert it to ms
% rate 20000 -> seconds = divide by sampling rate
% -> ms = 10^-6 = 10^-3/2
test = spikeVector(folder,0);
%test(:,2) = test(:,1)/2;


%freq_whole = sum(test(:,1))/((test(end,2)-test(1,2))*10




myKsDir = 'C:\Users\Nutzer\Google Drive\1 Uni\1.3 Uni Projekte\Masterarbeit Laborarbeit Neuroscience\Data Analysis\recordings\JG14_190621\electrophysiology';
sp = loadKSdir(myKsDir)
 
% plot a driftmap
[spikeTimes, spikeAmps, spikeDepths, spikeSites] = ksDriftmap(myKsDir);
figure; plotDriftmap(spikeTimes, spikeAmps, spikeDepths);

depthBins = 0:40:3840;
ampBins = 0:30:min(max(spikeAmps),800);
recordingDur = sp.st(end);

% not plotting correctly the amplitude?
[pdfs, cdfs] = computeWFampsOverDepth(spikeAmps, spikeDepths, ampBins, depthBins, recordingDur);
plotWFampCDFs(pdfs, cdfs, ampBins, depthBins);

% histogram of amplitude of spikes clusters




