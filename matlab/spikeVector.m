function spike_vec = spikeVector(folder,cluster)
    % load spike times & clusters & manual clusters
    % folder = dir with all kilosort % phy output files -> string in sampling
    % cluster = number of which luster to get spike vector
    
    if (exist(fullfile(folder,'spike_times.npy'))~=2) || (exist(fullfile(folder,'spike_clusters.npy'))~=2) || (exist(fullfile(folder,'cluster_group.tsv'))~=2) 
        fprintf("the files are not in the folder or wrong folder selected\n");
        return
    end
    
    if ~isnumeric(cluster) && ~(cluster >= 0)
        fprintf('class is not a number or greater equal 0\n')
        return
    end
    
    spike_times = readNPY(fullfile(folder, 'spike_times.npy'));
    spike_clusters = readNPY(fullfile(folder, 'spike_clusters.npy'));
    
    if size(spike_clusters) ~= size(spike_times)
        fprintf('there was an error: spike times not same size as clusters\n')
        return
    end
    
    cluster_group = tdfread(fullfile(folder, 'cluster_group.tsv'));
    spike_vec = spike_times(spike_clusters(:,1)==cluster,1);
    
    
    
    % number of neurosn firing together
    same_fiering = size(spike_times,1)-size(unique(spike_times),1);
    
end