function contourFrequencyPlot(clusters, trial_info, folder, x, level)

% prepare x, y, z axis
X = (1:size(clusters));
Y = trial_info.working_trials(:,1);

Z = [];
for cluster = clusters'
    name = strcat( 'cluster', num2str(cluster) );
    Z(:,end+1) = ( trial_info.spikes.(name)(trial_info.working_trials_bool(:,1)==1 ,4) );
end
Z(:,find( sum( Z(:,:) ) >300 ))=[];
%Z(:,find( sum( Z(:,:)==0 ) >40 ))=[];
Z(:,find( sum( Z(:,:) ) < 70 ))=[];
X = 1:size(Z,2);

%find(sum(Z(:,:)==0)>80)

%empty_in_row = sum(Z(:,:)==0)
%empty_in_row_tabluate = tabulate(empty_in_row)
%z_sum = sum(Z(:,:))
%find(sum(Z(:,:))>300)

%histogram(z_sum)
%z_sum_tabulate = tabulate(z_sum)

% plot contourf
f = figure(); 
clf;

%contourf(X,Y,Z, level, 'linecolor','none')
imagesc(X,Y,Z)
%set(gca,'xlim',[0 1.3])
title('Frequency in hz')
xlabel('Cluster'), ylabel('Trial')
colormap jet;
colorbar
%f.YDir='normal';

% plot prob change
gambl_change  = gamblProbChange(folder, trial_info, 1);
gambl_change = gambl_change(:,2);
for change = gambl_change'
    yline( change, '--r','LineWidth',1)
end



end