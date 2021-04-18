function createCheckpointFile(net,options,epoch,iteration)
chkFilename = fullfile(pwd,options.CheckpointPath,['trained_' num2str(epoch) '_' num2str(iteration)]);
if ~exist(fullfile(pwd,options.CheckpointPath),'dir')
    mkdir(fullfile(pwd,options.CheckpointPath))
end
save(chkFilename,'net','options');
end