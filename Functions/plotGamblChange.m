function plotGamblChange(folder,trial_info, work_bool)

% plot behavior change 
gambl_change  = gamblProbChange(folder, trial_info, 1);
gambl_change = gambl_change(:,2);
for change = gambl_change'
    xline( change, '--r',{'behaviro change'},'LineWidth',1)
end

end