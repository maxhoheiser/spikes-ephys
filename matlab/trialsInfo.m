function trial_info=trialsInfo(behavior_vec, folder)
% input <- BEHAVIOR_VEC & FOLDER
% 1. find each whole trial in behavior vec
% 2. index trials
% 3. compute trial length 
% 4. compute standart diviation
% 5. compute mean
% 6. compute firering rate for each trial
% 7. compute frequency for each trial
%
% ####store -> struct(trials_info)
trial_info = struct();
%%%%%%%%%%%%%%%%%
spike_times = readNPY(fullfile(folder, 'spike_times.npy'));
spike_clusters = readNPY(fullfile(folder, 'spike_clusters.npy'));
cluster_group = tdfread(fullfile(folder, 'cluster_group.tsv'));


% ####store -> list(trials)
% trial index, start index of behavior_vec, trial length, fireing count,
% frequency, 
%
% 1, 2, 3
trials(:,2) = find(behavior_vec(:,1)==1);
trials(:,1) = 1:size(trials,1);
trials(:,3) = behavior_vec(trials(:,2)+6,2) - behavior_vec(trials(:,2),2);
trial_info.trials = trials; 

% ####store -> struct(spikes) 
% trial index, firing rate for clustr 1, firing count cluster 2 ....
% 6, 7
trial_info.spikes = struct();
trial_info.spikes.names = ["trial num", "spike count", "spike rate per sampling-rate", "frequency"];
for cluster = cluster_group.cluster_id'
    spike_vec = spikeVector(folder,cluster);
    test = trials(:,2);
    spike_count = [];
    for trial_i = test'
        [interim_behavior , interim_spike] = spikePerTrial(behavior_vec, spike_vec, trial_i);
        spike_count(end+1,[2,3]) = [size(interim_spike,1), interim_behavior(7,3)];
    end
    clear('interim');
    spike_count(:,1) = 1:size(test,1);
    % calculate frequency per sampling rate
    spike_count(:,4) = spike_count(:,2)./trials(:,3);
    % calculating frequency
    spike_count(:,5) = spike_count(:,4)*2000;
    % store in substructure
    name = convertStringsToChars("cluster"+num2str(cluster));
    trial_info.spikes.(name)(:,[2,3,4]) = spike_count(:,[2,4,5]);
    trial_info.spikes.(name)(:,[1]) = trials(:,1);
end

% 4, 5
trial_info.stats = struct();
trial_info.stats.mean_trial_length = mean(trials(:,3));
trial_info.stats.std_trial_length = std(trials(:,3));

for cluster = cluster_group.cluster_id'
    name = convertStringsToChars("cluster"+num2str(cluster));
    trial_info.stats.mean_frequency.(name) = mean(trial_info.spikes.(name)(:,[2,3]));
    trial_info.stats.std_frequency.(name) = std(trial_info.spikes.(name)(:,[2,3]));
end


end