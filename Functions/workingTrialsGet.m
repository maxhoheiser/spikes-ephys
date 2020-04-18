function [working_trials, working_trials_bool] =workingTrialsGet(trials, threshold)
%std_range = 2;
%working_trials = trial_info.trials( ( (abs(trial_info.trials(:,3))-abs(trial_info.stats.mean_trial_length)) <= std_range*trial_info.stats.std_trial_length ), :);

working_trials = trials( abs(trials(:,3)) <= threshold , :);
working_trials_bool = abs(trials(:,3)) <= threshold ;

end