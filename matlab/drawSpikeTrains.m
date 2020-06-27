function drawSpikeTrains(behavior_vec,cluster,folder,trial_info,draw,y_step)
% draw spike trains for selected CLUSTER and selected TRIALS
% draw = FALSE -> save file with name = spiketrian_clusterID 
% draw = TRUE -> draw spiketriain in figure
%
trials = trial_info.working_trials;
spike_vec = spikeVector(folder,cluster);
%plott = figure(3), clf
%hold all
if nargin == 5
    y_step = 0.4;
end

if nargin == 4
    y_step = 0.4;
    draw = True;
end

trial_count = 0;

if draw
    figure;
    clf;
end
if ~draw
    f = figure('visible','off');
    clf;
end
for trial = trials(:,2)'
    [current_behavior_vec, current_spike_vec] = spikePerTrial(behavior_vec, spike_vec, trial);
    
    hold all;
    plot([current_behavior_vec(1,3) current_behavior_vec(1,3)], [trial_count-y_step trial_count+y_step],'-g');
    plot([current_behavior_vec(7,3) current_behavior_vec(7,3)], [trial_count-y_step trial_count+y_step],'-b');
    for plot_i = 1:size(current_spike_vec)
        plot([current_spike_vec(plot_i,2) current_spike_vec(plot_i,2)], [trial_count-y_step trial_count+y_step],'-k', 'LineWidth', 1);
    end
    
    trial_count = trial_count + 1;
end

ylim([0,trial_count]);

% draw gambling probability change
% plot behavior change 
gambl_change  = gamblProbChange(folder, trial_info, 1);
gambl_change = gambl_change(:,2);
for change = gambl_change'
    yline( change, '--r',{'change'},'LineWidth',1)
end

% store file if draw = False
if ~draw
    folder_fig = strcat( fileparts(folder), '\figures\spikes') ;   
    if ~exist(folder_fig, 'dir')
        mkdir(folder_fig)
    end
    folder_fig_train = strcat( folder_fig, '\spike-trains') ;
    if ~exist(folder_fig_train, 'dir')
        mkdir(folder_fig_train)
    end
    name = strcat( folder_fig_train, '\cluster_', num2str(cluster) );
    saveas(f,name,'fig');
    saveas(f,name,'jpeg');
%    saveas(f,name,'svg');
end

end