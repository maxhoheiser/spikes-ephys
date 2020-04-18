function list_change  = gamblProbChange(folder, trial_info, work_bool)
% find where the reward for gambling arm changes and map this to the
% work_bool = BOOLEAN -> TRUE = return trials + working trials change
% FAKSE -> return only trials change
% reduced trial vector and to trial vector
% plot change in reward percentage & name the bins
%
% read excel file in folder with michaels functions
subfolder = dir( strcat( fileparts(folder), '\behavior')  );
subfolder_content = dir(  strcat( subfolder(3).folder, '\', subfolder(3).name ) );
load(  strcat( subfolder(3).folder,'\',subfolder(3).name,'\',subfolder_content(4).name ) );

gambl_change = BehTrials.GambleProbChanges2([1,2]);

% map changes to working trials, check which trials were excluded and then
% adapt the starting point of change

gambl_change_mapped = [];
for change_trial = gambl_change'
    gambl_change_mapped(end+1,1) = find( trial_info.working_trials(:,1) == change_trial);
end


% [gambl_change_mapped, gambl_change]
if ~work_bool
    list_change = gambl_change;
elseif work_bool
    list_change = [gambl_change, gambl_change_mapped];
else
    fprintf("there was an error with the input arguments")
    return
end