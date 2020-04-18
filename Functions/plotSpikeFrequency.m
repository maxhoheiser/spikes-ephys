function plotSpikeFrequency(cluster, trial_info, draw, folder)


% plot linediagram for frequency of given trial
name = strcat( 'cluster', num2str(cluster) );
if draw
    f = figure;
    clf;
end
if ~draw
    f = figure('visible','off');
    clf;
end
plot( trial_info.spikes.(name)(trial_info.working_trials_bool,1), trial_info.spikes.(name)(trial_info.working_trials_bool,4) )
yl = ylim;
xlim([0,size(trial_info.spikes.(name)(trial_info.working_trials_bool,1),1)]);
xlabel('Trial')
ylabel('Frequency')
title( strcat( "Frequencyplot for Cluster ", num2str(cluster) ) );


% plot behavior change 
gambl_change  = gamblProbChange(folder, trial_info, 1);
gambl_change = gambl_change(:,2);
for change = gambl_change'
    xline( change, '--r','LineWidth',1)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% store file if draw = False
if ~draw
    folder_fig = strcat( fileparts(folder), '\figures\spikes') ;
    if ~exist(folder_fig, 'dir')
        mkdir(folder_fig)
    end
    folder_fig_train = strcat( folder_fig, '\spike-frequency') ;
    if ~exist(folder_fig_train, 'dir')
        mkdir(folder_fig_train)
    end
    name = strcat( folder_fig_train, '\cluster_', num2str(cluster) );
    saveas(f,name,'fig');
    saveas(f,name,'jpeg');
    %    saveas(f,name,'svg');
end

end