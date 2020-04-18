function [current_behavior_vec, current_spike_vec] = spikePerTrial(behavior_vec, spike_vec, trial_i)
% for trial TRIAL_I 
% get CURRENT_BEHAVIRO_VEC with times and beaviors of current trial 
% get CURRENT_SPIKE_VEC all spikes during that trial
% create subvector of trial
current_behavior_vec = [];
current_behavior_vec = behavior_vec(trial_i:trial_i+6,:);
% find spikes in timeframe of this trial
%# replace 1 by percistant
%# index which is the index of the last spike trial -> create a case
%# for first loop
current_spike_vec = [];
current_spike_vec = spike_vec((spike_vec(:,1)>=current_behavior_vec(1,2) & spike_vec(:,1)<=current_behavior_vec(7,2)),1);

% nomalize time in trial
current_behavior_vec(:,3) = current_behavior_vec(:,2) - current_behavior_vec(1,2);
current_spike_vec(:,2) = current_spike_vec(:,1) - current_behavior_vec(1,2);


end