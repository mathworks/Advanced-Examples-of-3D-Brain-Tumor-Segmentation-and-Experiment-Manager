%
% Copyright 2021 The MathWorks, Inc.

function dataAvailable = processValidation3dBrain(validationQ,displayQ,maxPollTime)
global g_clientValidationResponseSignal

if isa(validationQ,'parallel.pool.PollableDataQueue')
    [validationData,dataAvailable] = poll(validationQ,maxPollTime);
else
    dataAvailable = true;
    displayQ = [];
    validationData = validationQ;
end

% don't continue if 'stop_training' already exists.  stop_training file is
% deleted before final validataion so it can be ran one last time
assert(~exist('stop_training','file'), ...
    "backgroundProcesses3dBrain:processValidation3dBrain:validateModel3dbrain", ...
    "stop-training detected during validation.");
        
if dataAvailable && ~exist('stop_training','file')
    try
        net = validationData{1};
        options = validationData{2};
        pixelLabelID = validationData{3};
        classNames = validationData{4};
        epoch = validationData{5};
        iteration = validationData{6};
        dsValCombined = validationData{7};
        executionEnvironment = validationData{8};
        firstValidation = validationData{9};
        batchPerGPU = validationData{10};
        
        if ~isempty(options.CheckpointPath)
            createCheckpointFile(net,options,epoch,iteration);
        end
    catch ME
        data = [{[]} {[]} iteration {[]} {[]} {[]} {0} {0} {[]} {0}];
        send(g_clientValidationResponseSignal,data);
        rethrow(ME)
    end
    
    %     displayTrainingProgress({"validation_started"});
    startTime = tic;
    try
        [valAccur,valLoss,montageSlices] = validateModel3dBrain(net,dsValCombined,pixelLabelID,classNames,firstValidation,displayQ,batchPerGPU);
        valElapsedTime = floor(toc(startTime));
        if ~exist(fullfile(pwd,options.CheckpointPath),'dir')
            mkdir(fullfile(pwd,options.CheckpointPath));
        end
        montageFileName = fullfile(pwd,options.CheckpointPath,sprintf("validationMontage_%d.jpg",iteration));
        montageSize = size(montageSlices);
        im = [];
        %     for i=1:montageSize(4)
        %         imRow = cat(2,montageSlices(:,:,1,i),montageSlices(:,:,2,i));
        %         im = cat(1,im,imRow);
        %     end
        slicesPerRow = 1;
        for i=1:slicesPerRow:montageSize(4)
            Tset = reshape(montageSlices(:,:,1,(i-1)+(1:slicesPerRow)),montageSize(1),slicesPerRow*montageSize(2));
            Yset = reshape(montageSlices(:,:,2,(i-1)+(1:slicesPerRow)),montageSize(1),slicesPerRow*montageSize(2));
            imRow = cat(2,Tset,Yset);
            im = cat(1,im,imRow);
        end
        imR = double((im=="tumor"));
        imG = double((im=="background"));
        imB = zeros(size(imR),'like',imR);
        im=cat(3,imR,imG,imB);
        imWidth = montageSize(2);
        im(:,1:imWidth,:) = .333 * im(:,1:imWidth,:);
        im(:,(imWidth+1):end,:) = im(:,1:imWidth,:) + .666*im(:,(imWidth+1):end,:);
    catch ME
        data = [{[]} {[]} iteration {[]} {[]} {[]} {0 } {0} {[]} {0}];
        send(g_clientValidationResponseSignal,data);
        rethrow(ME)
    end
    
    try
        imwrite(im,montageFileName);
        data = [{[]} ...
            {[]} ...
            iteration ...
            {[]} ...
            {[]} ...
            {[]} ...
            {valAccur} ...
            {valLoss} ...
            {montageFileName} ...
            {valElapsedTime}];
        %         {montageSlices}];
        %         {fullfile(pwd,options.CheckpointPath)}];
        %     displayTrainingProgress({"validation_stopped"});
        %     displayTrainingProgress(data);
        %     figFilename = fullfile(pwd,options.CheckpointPath,['plot_' num2str(epoch) '_' num2str(iteration)]);
        validationLogFileName = fullfile(pwd,options.CheckpointPath,"validationLog.txt");
        if firstValidation
            validationLogFile = fopen(validationLogFileName,"w");
        else
            validationLogFile = fopen(validationLogFileName,"a");
        end
        fprintf(validationLogFile,"%s iteration=%d gpumem=%e valAccur_background=%f valAccur_normal=%f valLoss=%f,elapsedTime=%d\r\n",...
            timeofday(datetime),iteration,gpuDevice().AvailableMemory,valAccur(1),valAccur(2),valLoss,valElapsedTime);
        fclose(validationLogFile);
        send(g_clientValidationResponseSignal,data);
    catch ME
        data = [{[]} {[]} iteration {[]} {[]} {[]} {0} {0} {[]} {0}];
        send(g_clientValidationResponseSignal,data);
        rethrow(ME)
    end
    %     displayTrainingProgress([{"saveFig"} {figFilename}]);
    %     isBusy = false;
    % else
    %     ['isBusy ' num2str(validationData{4})]
    % end
end
end

