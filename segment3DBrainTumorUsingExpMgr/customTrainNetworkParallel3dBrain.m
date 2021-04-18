% This is an implementation of custom DL training using dlnetwork(),
% dlarray(), dlfeval and dlupdate.   Several novel implementations include
% * supporting batchNormalizationLayer (BN) in original mode and in custom mode.
%       original mode (how handled using trainNetwork()): learns normalization
%       scale and offset using current iteration's minibatch statistics.
%       Final statistical mean and variance based on full batch data.
%       Hence, normal training is followed by an additional epoch just for
%       determining final statistics.
%
%       custom training (a.k.a. "Functional") mode: final statistical mean
%       and variance based on running average of last 10 iterations. For
%       example, new running mean = 0.9*old running mean + 0.1 current
%       minibatch mean.
% * randomizing minibatches across multiple gpu's
%       the datastore is duplicated across gpu's.  Each epoch, a randomized
%       list of indices is regenerated and each gpu is assigned a portion
%       of the list so that the data cannot be duplicated across gpu's
% * extending miniBatchSizes to create virtual miniBatchSizes.
%       In order to facilitate limited memory or to emulate multiple
%       gpu's iteration data is aggregated across sub iterations, as well
%       as, multiple gpu's.
%
%
% Copyright 2021 The MathWorks, Inc.

function     [net,trainingInfo] = customTrainNetworkParallel3dBrain(...
    dsTrainCombined,dsValCombined,...
    lgraph,...
    options,...
    miniBatchMultiplier,...
    diceLayer,...
    bnUseRunningMean,...
    pixelLabelID, classNames, ...
    leaveOneOutIdx, hideRedundantInfo, expMonitor)

%%
% The general work flow for custom training is:
%
% 1. convert layer graph to dlnetwork object
% 2. implement a nested loop for number of epochs and
%    number of iterations.
% For each iteration:
% 3. get next minibatch of data
% 4. if using gpu, convert minibatch to gpuArray
% 5. use dlfeval to call custom function for processing forward and loss
% functions. dlfeval() returns gradients,loss and persistent states of
% layers, for example, BN running averages.
% 6. aggregate gradients, losses, states across gpu's.
% 6.5 if extended/virtual minibatch, aggregate each accross sub-iterations
% 7. call any gradient thresholding functions
% 8. call optimizer function to update learnables.
%

%%

%the existance of the file stop_training will stop the training in
%a soft manner, completing the current iteration and finalizing the
%training.
if exist('stop_training','file')
    delete stop_training
end

%%
% dlnetwork() requires final output processing to be done in the dlfeval()
% function, following the forward calculations.  Hence the end layers need
% to be removed from the layer array.

trimmedLGraph = removeLayers(lgraph,["Softmax-Layer" string(lgraph.OutputNames)]);
newEndLayerInx = find(strcmp(lgraph.Connections.Destination,"Softmax-Layer"));
newEndLayerName = lgraph.Connections.Source{newEndLayerInx};
trimmedLGraph = addLayers(trimmedLGraph,dropoutLayer(0,'Name','placeholder-end-layer'));
trimmedLGraph = connectLayers(trimmedLGraph,newEndLayerName,'placeholder-end-layer');

%use the following internal function to pass to dlfeval for calculating the
%loss.
% diceLayer = nnet.internal.cnn.layer.GeneralizedDiceLoss('diceLoss',categorical([]),[],4);

% convert layer graph to dlnetwork object
dlnetPerGPU = dlnetwork(trimmedLGraph);

% note: any table in the form of dlnet.Learnables or dlnet.States can be
% processed by dlupdate.  It will call a custom function for each rows
% Value fields and return the modified values. (dlupdate acts similar for
% others kinds of data).  For the unet network only BNs have
% states and I want to initialize the running averages to 0.  This is
% really only neccessary for processing the original mode.
dlnetPerGPU.State = dlupdate(@zeroState,dlnetPerGPU.State);

% Create DataQueue for sending data back from workers during training.
% when using spmd or parfor for parallel processing status reporting
% messages are asynchronouse and could be confusing.  DataQueues are used
% to facilitate reporting in parallel processing

if options.ExecutionEnvironment == "multi-gpu"
    backgroundProcessesQ = parallel.pool.DataQueue;
    afterEach(backgroundProcessesQ,@backgroundProcesses3dBrain);
    
    displayQ = parallel.pool.PollableDataQueue;
    validationQ = parallel.pool.PollableDataQueue;
elseif options.ExecutionEnvironment == "gpu"
    % The queues only facilitate parallel constructs, such as spmd and parfor.
    % if (single)gpu mode then a work-around is to disable the backroundProcess,
    % set the displaQ and validationQ as event based DataQueues and
    % wrap the queue sends in spmd blocks.
    displayQ = parallel.pool.DataQueue;
    afterEach(displayQ,@displayTrainingProgress3dBrain);
    validationQ = parallel.pool.DataQueue;
    afterEach(validationQ,@processValidation3dBrain);
end

% determine number of gpu.  only restart parpool if number of workers has
% changed.
numGPU = sum(floor(options.WorkerLoad));
if options.ExecutionEnvironment == "gpu"
    numGPU = 1;
end

if numGPU>0
    if ~isempty(gcp('nocreate'))
        if gcp('nocreate').NumWorkers ~= numGPU
            delete(gcp);
            parpool(numGPU);%+numel(validationWorker));
        end
    else
        parpool(numGPU);%+numel(validationWorker));
    end
else
    if ~isempty(gcp('nocreate'))
        if gcp('nocreate').NumWorkers ~= 1
            delete(gcp);
            parpool(1);
        end
    else
        parpool(1);
    end
end

if numGPU >0
    % The signals only facilitate parallel constructs, such as spmd and parfor.
    % if (single)gpu mode then a work-around is to
    % wrap the signal queries in spmd blocks.
    spmd
        workerStopTrainingSignal = [];
        validationResponseSignal = [];
        if labindex==1
            workerStopTrainingSignal = parallel.pool.PollableDataQueue;
            validationResponseSignal = parallel.pool.PollableDataQueue;
        end
    end
end
global  g_clientStopTrainingSignal g_clientValidationResponseSignal 
g_clientStopTrainingSignal = workerStopTrainingSignal{1};
g_clientValidationResponseSignal = validationResponseSignal{1};

executionEnvironment = options.ExecutionEnvironment;

% N represents number of workers/gpu for purposes of distributing
% datastore across workers
if executionEnvironment == "multi-gpu"
    N = sum(options.WorkerLoad);
else
    N = 1;
end

global g_expMonitor
g_expMonitor = expMonitor;

%sometimes I need to reset the gpu's
%don't know if this is really neccessary
spmd
    reset(gpuDevice);
end

% The spmd block of code is run on each gpu.  labindex value indicates the
% respective gpu number.  WorkerLoad indicates which gpu's will be used for
% the training

% Any variables assigned in the blocks become composite variables.  After
% parallel processing variables for each block can be extracted using the
% variable name with {gpu number}.  For example, to extract dlnet generated
% from gpu number 2,  it woulbe referenced dlnet{2}.
if executionEnvironment == "multi-gpu"
    spmd
        % only train if indicated by WorkerLoad
        if options.WorkerLoad(labindex) || numGPU==0
            if labindex==1
                send(backgroundProcessesQ,[{displayQ},{validationQ}]);
                send(displayQ,[]);
                if hideRedundantInfo
                    send(displayQ,[{"hideRedundantInfo"}]);
                end
                send(displayQ,[{"figure_name"},{pwd}]);
            end
            [net,  trainingInfo] = trainDLNetwork(...
                dlnetPerGPU, ...
                executionEnvironment,...
                options,...
                pixelLabelID, classNames, ...
                dsTrainCombined, dsValCombined, ...
                diceLayer,...
                N,...
                miniBatchMultiplier, ...
                bnUseRunningMean,...
                displayQ,validationQ,...
                workerStopTrainingSignal,validationResponseSignal,...
                leaveOneOutIdx,expMonitor);
            if labindex==1
                send(displayQ,[{"Stop backgroundProcesses"}]);
            end
        end
    end %spmd
elseif executionEnvironment == "gpu"
    spmd
        send(displayQ,[]);
        if hideRedundantInfo
            send(displayQ,[{"hideRedundantInfo"}]);
        end
        send(displayQ,[{"figure_name"},{pwd}]);
%         send(displayQ,[{"initialize_expMonitor"} {expMonitor}]);
    end
    [net,  trainingInfo] = trainDLNetwork(...
        dlnetPerGPU, ...
        executionEnvironment,...
        options,...
        pixelLabelID, classNames, ...
        dsTrainCombined, dsValCombined, ...
        diceLayer,...
        N,...
        miniBatchMultiplier, ...
        bnUseRunningMean,...
        displayQ,validationQ,...
        workerStopTrainingSignal,validationResponseSignal, ...
        leaveOneOutIdx,expMonitor);
    
    spmd
        send(displayQ,[{"Stop backgroundProcesses"}]);
    end
end

if isa(net,"Composite")
    net = net{1};
end
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,'placeholder-end-layer');
lgraph = connectLayers(lgraph,newEndLayerName,"softmax");
net = assembleNetwork(lgraph);

if isa(trainingInfo,"Composite")
    trainingInfo = trainingInfo{1};
end

if executionEnvironment == "multi-gpu"
    send(backgroundProcessesQ,false);
    pause(5);
    delete(backgroundProcessesQ);
end
delete(displayQ);
delete(validationQ);


spmd
    reset(gpuDevice); %required to clear memory from predictUnet()
end

end

%%
function [net,  trainingInfo] = trainDLNetwork(...
    dlnetPerGPU,...
    executionEnvironment,...
    options,...
    pixelLabelID, classNames, ...
    dsTrainCombined, dsValCombined, ...
    diceLayer,...
    N,...
    miniBatchMultiplier, ...
    bnUseRunningMean, ...
    Q,validationQ,...
    stopTrainingSignal,validationResponseSignal, ...
    leaveOneOutIdx,expMonitor)

global g_hardStop g_endOfIterationStop g_endOfEpochStop %g_expMonitor

% g_expMonitor = expMonitor;
g_hardStop = false;
g_endOfIterationStop = false;
g_endOfEpochStop = false;
valBusy = false;
lastValidationTime = tic;
minTimeBetwVal = 0;
encodedFreq = typecast(uint64(options.ValidationFrequency),'int64');
if  encodedFreq < 0
    encodedFreq = abs(encodedFreq);
    minTimeBetwVal = encodedFreq * 60;
end

% to track performance
start = tic();

firstValidation = true;

%extract maximum epochs and generate a learnrate array
numEpochs = options.MaxEpochs;

if options.LearnRateSchedule == "piecewise"
    learnRatePerEpoch = zeros(1,numEpochs);
    for e=1:options.LearnRateDropPeriod:numEpochs
        learnRatePerEpoch((e-1)+(1:options.LearnRateDropPeriod))=...
            options.InitialLearnRate * options.LearnRateDropFactor^floor(e/options.LearnRateDropPeriod);
    end
elseif options.LearnRateSchedule == "none"
    learnRatePerEpoch = repmat(options.InitialLearnRate,1,numEpochs);
end
trainingInfo.BaseLearnRate = learnRatePerEpoch;

% determine minibatchsize and size-per-gpu, etc.
% subIterations is used for extending the minibatch size beyond memory
% limits.  Hence, losses and gradients are aggregated over multiple
% sub-iterations, then optimization only happens after a complete
% iteration.
miniBatchSize = options.MiniBatchSize;
batchPerGPU = floor(miniBatchSize/N);
miniBatchSize = miniBatchSize * miniBatchMultiplier;
if any(contains(properties(dsTrainCombined),"UnderlyingDatastores"))
    %non-blockedDatastore
    iterationsPerBatch = ceil(numel(dsTrainCombined.UnderlyingDatastores{1}.Files)/miniBatchSize);
else
    % blockedDatastore
    iterationsPerBatch = ceil(dsTrainCombined.TotalNumBlocks/miniBatchSize);
end
subIterationsPerMiniBatch = ceil(miniBatchSize/(N*batchPerGPU));

mbq = minibatchqueue(dsTrainCombined, ...
    'MiniBatchSize', batchPerGPU,...
    'MiniBatchFcn',@buildBatch, ...
    'MiniBatchFormat', [{'SSSCB'},{'SSSCB'}]);


% initialize adam optimizer persistent variables
workerAverageGrad = [];
workerAverageSqGrad = [];

% split images across multiple gpu's
if any(contains(properties(dsTrainCombined),"UnderlyingDatastores"))
    %non-blockedDatastore
    nPerGPU = floor(numel(dsTrainCombined.UnderlyingDatastores{1}.Files)/N);
else
    %blockedDatastore
    nPerGPU = floor(dsTrainCombined.TotalNumBlocks/N);
end

% To support BatchNormalization-original mode,
% establish arrays to hold state values for each iteration for last
% batch.
workerStatePerMiniBatch = cell(iterationsPerBatch,2);
workerStatePerSubMiniBatch = cell(subIterationsPerMiniBatch,2);

iteration = 1;
% add extra epoch if need to handle batchNormalization final run
maxEpochs = options.MaxEpochs+1;
if labindex==1
    logFile = fopen("logFile.txt","w");
    fclose(logFile);
end
for epoch = 1:maxEpochs
    % to report performance
    if labindex == 1
        logFile = fopen("logFile.txt","a");
        epochStart = tic;
    end
    
    % to support aggregating minibatches over sub-iterations need to
    % reset some persistent aggregating variables
    miniBatchObsCtr = 0;
    miniBatchGradients = dlupdate(@zeroState,dlnetPerGPU.Learnables);
    miniBatchLoss = 0;
    
    % Shuffling order of data: only want one gpu to generate random
    % indices and set the other gpu shuffle indices to zero.  With
    % gplus to add indice arrays across gpu's they will duplicate
    % indice arrays.
    if labindex==1
        shuffle(mbq);
    else
        mbq = [];
    end
    mbqEpoch = gcat({mbq});
    mbqEpoch = mbqEpoch{1};
    
    % use respective indice to partition the datastore
    mbqPartition = partition(mbqEpoch,N,labindex);
    
    %setup dataloader
    %         loader = nnet.internal.cnn.DataLoader(dsPartition,'MiniBatchSize',batchPerGPU,'CollateFcn',@buildBatch);
    sizeFirstSubMiniBatch = 0;
    subIteration = 0;
    iterationPerEpoch = 1;
    % Loop over mini-batches.
    while true

        send2Q(Q,executionEnvironment,[{sprintf("Running Iteration: %d. elapsed time: %s",iteration, duration(0,0,toc(start),'Format','hh:mm:ss'))} {2}]);
        if labindex == 1
            fprintf(logFile,"Running Iteration: %d. elapsed time: %s\r\n",iteration, duration(0,0,toc(start),'Format','hh:mm:ss'));
        end
        
        
        % epoch done when loader has no more data
        % gop confirms that all gpu's have exhausted their data.
        % it is possible to have uneven data distribution across
        % gpu's
        pollForStopFlags(stopTrainingSignal,executionEnvironment);
        if ~gop(@and,hasdata(mbqPartition)) || g_hardStop
            break;
        end
        
        subIteration = subIteration + 1;
        
        % Form next batch for input and output variables
        %             batch = nextBatch(loader);
        [X1,Y] = next(mbqPartition);
        
        if leaveOneOutIdx > 0 
            %zero respective channel to keep it from contributing
            X1(:,:,:,leaveOneOutIdx,:) = 0;
        end
        
        % if using gpu, then cast data as gpuArray
        %             if contains(executionEnvironment,"gpu")
        %                 batch = dlupdate(@gpuArray,batch);
        %             end
        
        %             [X1,Y] = batch{:};
        sizeX = size(X1);
        sizeMiniBatchGPU = sizeX(end);
        
        % need to know sizes across gpu's. again, it is possible for
        % uneven distribution.
        sizeMiniBatchAllGPU = gplus(sizeMiniBatchGPU);
        if sizeFirstSubMiniBatch==0
            sizeFirstSubMiniBatch = sizeMiniBatchAllGPU;
        end
        
        % because it is possible to have uneven distribution across
        % the gpu's and sub-iterations we need to determine
        % weighting factor as proportion of total contribution for
        % aggregation purposses.
        workerNormalizationFactor = sizeMiniBatchGPU./miniBatchSize;
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.  The results will be different
        % for each gpu, so they will then have to be aggregated.
        pollForStopFlags(stopTrainingSignal,executionEnvironment);
        if g_hardStop
            break;
        end
        [workerGradients,dlworkerLoss,workerState] = dlfeval(@modelGradients,dlnetPerGPU,X1,diceLayer,Y);
        
        
        % for managing BN in the original mode, we
        % must save the statistics.  This step isn't required for
        % the custom mode.  Save the data size, too, to use for
        % calculating the weighting factor.
        [workerState2,sizeMiniBatch] = extractBNMiniBatchMeansAndVars(workerState, sizeMiniBatchGPU);
        workerStatePerSubMiniBatch(subIteration,:) = [{workerState2} {sizeMiniBatch}];
        
        pollForStopFlags(stopTrainingSignal,executionEnvironment);
        if g_hardStop
            break;
        end
        
        % for normal training epochs, aggregate the losses and
        % adjust the gradients.  If using BN original mode, the
        % following should not be done during the finalization
        % (last) epoch
        if epoch <= options.MaxEpochs
            % Aggregate the losses on all workers.
            miniBatchLoss = miniBatchLoss + gplus(workerNormalizationFactor*extractdata(dlworkerLoss));
            
            %L2 regularization
            %During training, after computing the model gradients, for each of the weight parameters,
            %add the product of the L2 regularization factor and the weights to the computed gradients
            %using the dlupdate function. To update only the weight parameters, extract the parameters
            %with name "Weights".
            l2RegularizationFactor = options.L2Regularization;
            idx = dlnetPerGPU.Learnables.Parameter == "Weights";
            workerGradients.Value(idx,:) = dlupdate(@l2Regularization , workerGradients.Value(idx,:), dlnetPerGPU.Learnables.Value(idx,:),{l2RegularizationFactor});
            
            %Gradient Clipping...similar to L2 regularization
            gradientThreshold = options.GradientThreshold;
            if ~isinf(gradientThreshold)
                workerGradients.Value = dlupdate(@thresholdL2Norm,workerGradients.Value,{gradientThreshold});
            end
            
            % Aggregate the gradients over sub-iterations.
            miniBatchGradients = dlupdate(@aggregateGradientsOverSubiterations,miniBatchGradients,workerGradients);
        end
        
        %accumulate observation counter
        miniBatchObsCtr = miniBatchObsCtr + sizeMiniBatchAllGPU;
        
        pollForStopFlags(stopTrainingSignal,executionEnvironment);
        if g_hardStop
            break;
        end
        
        % if acheived complete iteration then aggregate the BN
        % states, aggregate gradients over workers/gpu's
        if miniBatchObsCtr >= miniBatchSize
            [workerStateTemp,sizeMiniBatch] = aggregateStatesOverSubIterations(workerStatePerSubMiniBatch);
            workerStatePerMiniBatch(iterationPerEpoch,:) = [{workerStateTemp},{sizeMiniBatch}];
            
            if epoch <= options.MaxEpochs
                miniBatchGradients.Value = dlupdate(@aggregateGradientsOverWorkers,miniBatchGradients.Value,{workerNormalizationFactor});
                % Determine learning rate for time-based decay learning rate schedule.
                learnRate = learnRatePerEpoch(epoch);
                if bnUseRunningMean
                    dlnetPerGPU.State = workerStateTemp;
                end
                
                % Update the network parameters using the ADAM optimizer.
                [dlnetPerGPU.Learnables,workerAverageGrad,workerAverageSqGrad] = adamupdate(dlnetPerGPU.Learnables,miniBatchGradients,workerAverageGrad,workerAverageSqGrad,iterationPerEpoch,learnRate);
                
                valAccur = [];
                valLoss = [];
                if epoch > 1
                    if labindex == 1 && ...
                            ((minTimeBetwVal > 0 && toc(lastValidationTime)>minTimeBetwVal && ~mod(iteration,options.VerboseFrequency)) || ...
                            (minTimeBetwVal == 0 && ~mod(iteration,options.ValidationFrequency)))
                        send2Q(Q,executionEnvironment,[{"Validating..."} {1}]);
                        fprintf(logFile,"Validating...gpumem=%e\r\n",gpuDevice().AvailableMemory);
                        tempDLNet = dlnetPerGPU;
                        if ~bnUseRunningMean
                            tempDLNet = finalizeTraining(tempDLNet,workerStatePerMiniBatch);
                        end
                        net = dl2Net(tempDLNet, ...
                            [softmaxLayer("Name","softmax");...
                            dicePixelClassificationLayer("Name","dice-pixel-class", ...
                            "Classes",diceLayer.ClassNames)]...
                            );
                        if ~exist( 'validationQ','var')
                            if ~isempty(options.CheckpointPath)
                                createCheckpointFile(net,options,epoch,iteration);
                            end
                            [valAccur,valLoss] = validateModel(net,dsValCombined,firstValidation);
                        else
                            send2Q(validationQ,executionEnvironment,[{net},{options},{pixelLabelID},{classNames},...
                                {epoch},{iteration},{dsValCombined},{"cpu"},{firstValidation},{batchPerGPU}]);
                            fprintf(logFile,"validationQ; net options %d %d %d\r\n",epoch,iteration,firstValidation);
                        end
                        
                        send2Q(Q,executionEnvironment,[{" "} {1}]);
                        fprintf(logFile," \r\n");
                        firstValidation = false;
                        valBusy = true;
                        valBusyTime = tic;
                        try
                            while valBusy
                                if mod(toc(valBusyTime),60)<9 %once per minute, otherwize may fill up queue
                                    send2Q(Q,executionEnvironment,[{sprintf("waiting for validation to complete %f\r\n",toc(valBusyTime))} {2}]);
                                    fprintf(logFile,"waiting for validation to complete %f\r\n",toc(valBusyTime));
                                end
                                [valIteration,valAccur,valLoss] = pollForValidationResponse(validationResponseSignal,executionEnvironment,logFile,Q,5);
%                                 pollForStopFlags(stopTrainingSignal,executionEnvironment,-1); %only checking stop_training file
                                if ~isempty(valIteration) || anyStopFlag
                                    send2Q(Q,executionEnvironment,[{sprintf("validation completed %f\r\n",toc(valBusyTime))} {2}]);
                                    fprintf(logFile,"validation completed %f\r\n",toc(valBusyTime));
                                    valBusy = false;
                                    lastValidationTime = tic;
                                end
                            end
                        catch ME
                            warning([ME.message ':' 'waiting for validation: wait will be terminated.'])
                            send2Q(Q,executionEnvironment,[{ME.message} {1}]);
                        end
                    end
                    labBarrier;
                end
                
                % report iteration results
                if labindex == 1 && (iteration==1 || mod(iteration,options.VerboseFrequency)==0)
                    data = [{toc(start)} ...
                        {epoch} ...
                        {iteration} ...
                        {miniBatchObsCtr} ...
                        {miniBatchLoss} ...
                        {learnRate} ...
                        {valAccur} ...
                        {valLoss} ];
                    send2Q(Q,executionEnvironment,data);
                    fprintf(logFile,"%s elapsed_time %d %d %d %f %e %f %f\r\n",timeofday(datetime),...
                        epoch,iteration,miniBatchObsCtr,miniBatchLoss,learnRate,valAccur,valLoss);
                end
            end
            
            % reset a bunch of stuff
            miniBatchObsCtr = 0;
            iterationPerEpoch = iterationPerEpoch + 1;
            iteration = iteration + 1;
            subIteration = 0;
            miniBatchGradients.Value = zeroGradients(dlnetPerGPU.Learnables.Value);
            miniBatchLoss = 0;
            pollForStopFlags(stopTrainingSignal,executionEnvironment);
            if g_hardStop || g_endOfIterationStop
                break;
            end
        end
        
        pollForStopFlags(stopTrainingSignal,executionEnvironment);
        if g_hardStop
            break;
        end
        
    end %while
    if labindex==1
        fclose(logFile);
    end
    pollForStopFlags(stopTrainingSignal,executionEnvironment);
    if anyStopFlag
        break;
    end
end %for epoch

% Finalize the training, i.e. last validation, finalize statistics of
% batchNormalization if not using Running Mean.
if labindex==1
    logFile = fopen("logFile.txt","a");
end
pollForStopFlags(stopTrainingSignal,executionEnvironment);

% remove stop_training file here because it may cause another trap by a
% final validation, where it is checked for
if labindex==1
    if exist('stop_training','file')
        delete stop_training
    end
end

net=[];
if labindex == 1
    if ~g_hardStop
        if epoch > 1
            send2Q(Q,executionEnvironment,[{"finalizing training."} {1}]);
            fprintf(logFile,"finalizing training.\r\n");
            tempDLNet = dlnetPerGPU;
            if ~bnUseRunningMean
                tempDLNet = finalizeTraining(tempDLNet,workerStatePerMiniBatch);
            end
            net = dl2Net(tempDLNet, ...
                [softmaxLayer("Name","softmax");...
                dicePixelClassificationLayer("Name","dice-pixel-class", ...
                "Classes",diceLayer.ClassNames)]...
                );
            if ~exist( 'validationQ','var')
                createCheckpointFile(net,options,epoch,iteration);
                [valAccur,valLoss,~] = validateModel(net,dsValCombined,firstValidation);
            else
                %if validation was running in background, wait for
                %completion
                if valBusy
                    send2Q(Q,executionEnvironment,{"Waiting for previous validation to finish..."});
                    for secs = 1:(5*60)
                        [valIteration,valAccur,valLoss] = pollForValidationResponse(validationResponseSignal,executionEnvironment,logFile,Q);
                        if ~isempty(valIteration)
                            send2Q(Q,executionEnvironment,{"Validation completed."});
                            break;
                        end
                    end
                end
                
                %final validation
                send2Q(validationQ,executionEnvironment,[{net},{options},{pixelLabelID},{classNames},...
                    {epoch},{iteration},{dsValCombined},{"gpu"},{firstValidation},{batchPerGPU}]);
                valIteration = [];
                send2Q(Q,executionEnvironment,{"Waiting for final validation..."});
                for secs = 1:(5*60)
                    [valIteration,valAccur,valLoss] = pollForValidationResponse(validationResponseSignal,executionEnvironment,logFile,Q);
                    if ~isempty(valIteration)
                        send2Q(Q,executionEnvironment,{"Validation completed."});
                        break;
                    end
                end
                if isempty(valIteration)
                    send2Q(Q,executionEnvironment,{"Validation did not complete in time."});
                end
                fprintf(logFile,"validationQ; net options %d %d \r\n",epoch,valIteration);
            end
            
            % save checkpoints
            if ~isempty(options.CheckpointPath)
                savePlots(options.CheckpointPath,executionEnvironment,epoch,iteration,Q)
                fprintf(logFile,"savePlot: %d d\r\n", epoch,iteration);
            end
            if g_endOfEpochStop || g_endOfIterationStop
                send2Q(Q,executionEnvironment,[{"Training Completed Early, initiated by user"} {1}]);
                fprintf(logFile,"Training Completed Early, initiated by user\r\n");
            else
                send2Q(Q,executionEnvironment,[{"Training Complete"} {1}]);
                fprintf(logFile,"Training Complete\r\n");
            end
        end
    else
        send2Q(Q,executionEnvironment,[{"Training Terminated by user."} {1}]);
        fprintf(logFile,"Training Terminated by user.\r\n");
    end
    if labindex==1
        fclose(logFile);
    end
end
end



%%
%%
function gradOut = zeroGradients(gradIn)
gradOut = gradIn;
for i=1:numel(gradIn)
    gradOut{i}(:) = 0;
end
end

function gradients = aggregateGradientsOverWorkers(dlgradients,factor)
gradients = extractdata(dlgradients);
if iscell(gradients)
    gradients = cell2mat(gradients);
end
gradients = gplus(factor*gradients,class(gradients));
end

function updatedGradients = aggregateGradientsOverSubiterations(runningGradients,newGradients)
updatedGradients = runningGradients + newGradients;
end

function [gradients,loss,state] = modelGradients(dlnet,X1,diceLayer,Y)

% state contains layer persistent data, For unet, only BN has such data
[dlYPred,state] = forward(dlnet,X1);
dlYPred = softmax(dlYPred);

loss = diceLayer.forwardLoss(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

end

% function batchOut = buildBatch(b)
% % Given numObservation x numVars cell array, form the next batch
%
% batchOut = cell(1,size(b,2));
% % dlarray requires to specify dimension type, 'SSSCB' defines dimensions as
% % Spacial-Spacial-Spacial-Channel-Batch(Observation)
% batchOut{1} = dlarray(single(cat(5,b{:,1})),'SSSCB');
% batchOut{2} = dlarray(single(nnet.internal.cnn.util.dummifyND(cat(5,b{:,2}),5)),'SSSCB');
% end
%
function [Xout, Yout] = buildBatch(Xin,Yin)
% Given numObservation x numVars cell array, form the next batch

% dlarray requires to specify dimension type, 'SSSCB' defines dimensions as
% Spacial-Spacial-Spacial-Channel-Batch(Observation)
Xout = cat(5,Xin{:,1});
Yout = onehotencode(cat(5,Yin{:,1}),4);
end

function gradients = thresholdL2Norm(gradients,gradientThreshold)

gradientNorm = sqrt(sum(gradients(:).^2));
if gradientNorm > gradientThreshold
    gradients = gradients * (gradientThreshold / gradientNorm);
end

end
function l2gradients = l2Regularization(gradients,weights,regulationFactor)

l2gradients = gradients + regulationFactor*weights;
end
function state = zeroState(state)
state = 0*state + eps(state);
end




%%
function [stateToSave,sampleSizeToSave] = extractBNMiniBatchMeansAndVars(dlnetStates, sampleSize)
%Each BN has a pair of persistent states: Mean and Variance
idxMean = dlnetStates.Parameter == "TrainedMean";
idxVar = dlnetStates.Parameter == "TrainedVariance";
stateToSave = dlnetStates;
%aggregate the states over the GPU's
[stateToSave.Value(idxMean,:), stateToSave.Value(idxVar,:),sampleSizeToSave] = ...
    dlupdate(@aggregateStatesOverGPU,dlnetStates.Value(idxMean,:),dlnetStates.Value(idxVar,:),{sampleSize});
sampleSizeToSave = sampleSizeToSave{1};
end
function [updatedStateMean,updatedStateVar,updatedSamples] = aggregateStatesOverGPU(stateMean, stateVar, sampleSizes)

% variance is the sum((X-mean)^2)/N
% an estimate could be sum(X^2-mean^2)/N.
% the below estimates for weighted averages is based on the above estimate.

sampleSizes = gop(@horzcat,{sampleSizes});
statesMean = gop(@horzcat,{stateMean});
statesVar = gop(@horzcat,{stateVar});
runningStateMean = statesMean{1};
runningStateVar = statesVar{1};
runningSamples = sampleSizes{1};
for i=2:numel(sampleSizes)
    factor = sampleSizes{i}/(runningSamples+sampleSizes{i});
    updatedStateMean = (1-factor)*runningStateMean + factor*statesMean{i};
    updatedStateVar = (1-factor)*(runningStateVar+runningStateMean.^2) ...
        + factor*(statesVar{i}+statesMean{i}.^2) ...
        - updatedStateMean.^2;
    runningSamples = runningSamples+sampleSizes{i};
    runningStateMean = updatedStateMean;
    runningStateVar = updatedStateVar;
end
updatedStateMean = runningStateMean;
updatedStateVar = runningStateVar;
updatedSamples = runningSamples;
end

function [updatedState, updatedSampleSize] = aggregateStatesOverSubIterations(statesAndSizes)
states = statesAndSizes(:,1);
sampleSizes = statesAndSizes(:,2);

for i=1:numel(states)
    idxMean = states{i}.Parameter == "TrainedMean";
    idxVar = states{i}.Parameter == "TrainedVariance";
    if i==1
        updatedState = states{i};
        updatedState.Value = dlupdate(@zeroState,updatedState.Value);
        updatedSampleSize = 0;
    end
    factor = sampleSizes{i}/(updatedSampleSize+sampleSizes{i});
    additionalState = states{i};
    [updatedState.Value(idxMean,:), updatedState.Value(idxVar,:)] = dlupdate(@updateStateMeanAndVar, ...
        updatedState.Value(idxMean,:),updatedState.Value(idxVar,:), ...
        additionalState.Value(idxMean,:),additionalState.Value(idxVar,:), {factor});
    updatedSampleSize = updatedSampleSize+sampleSizes{i};
end
end
function [updatedStateMean,updatedStateVar] = updateStateMeanAndVar(runningStateMean,runningStateVar,additionalStateMean,additionalStateVar,factor)
% variance is the sum((X-mean)^2)/N
% an estimate could be sum(X^2-mean^2)/N.
% the below estimates for weighted averages is based on the above estimate.
updatedStateMean = (1-factor)*runningStateMean + factor*additionalStateMean;
if factor ==1
    updatedStateVar = additionalStateVar;
else
    updatedStateVar = (1-factor)*(runningStateVar + runningStateMean.^2) ...
        + factor*(additionalStateVar + additionalStateMean.^2) ...
        - updatedStateMean.^2;
end
end

function dlnet = finalizeTraining(dlnet,workerState)

isComposite = isa(workerState,'Composite');
numWorkers=1;

if isComposite
    tempState = workerState{1};
    numWorkers=numel(workerState);
else
    tempState = workerState;
end
meanIdx = tempState{1}.Parameter == "TrainedMean";
varIdx = tempState{1}.Parameter == "TrainedVariance";
finalState = tempState{1,1};
finalState.Value(meanIdx,:) = dlupdate(@zeroState,finalState.Value(meanIdx,:));
finalState.Value(varIdx,:) = dlupdate(@zeroState,finalState.Value(varIdx,:));
runningNumObservations = 0;
numIterations = size(tempState,1);

for w=1:numWorkers
    for iteration=1:numIterations
        if isComposite
            tempState = workerState{w};
        else
            tempState = workerState;
        end
        if ~isempty(tempState{iteration,1})
            minibatchFinalState = tempState{iteration,1};
            miniBatchSize = tempState{iteration,2};
            runningNumObservations = runningNumObservations + miniBatchSize;
            [finalState.Value(meanIdx,:),finalState.Value(varIdx,:)] = dlupdate(@finalizeTrainMeanAndVar,finalState.Value(meanIdx,:),finalState.Value(varIdx,:),minibatchFinalState.Value(meanIdx,:),minibatchFinalState.Value(varIdx,:),{miniBatchSize/runningNumObservations});
        end
    end
end
dlnet.State = finalState;
end

function [updatedStateMean, updatedStateVar] = finalizeTrainMeanAndVar(runningStateMean,runningStateVar, additionalStateMean,additionalStateVar, factor)
% variance is the sum((X-mean)^2)/N
% an estimate could be sum(X^2-mean^2)/N.
% the below estimates for weighted averages is based on the above estimate.

% the BN states are actually 0.1 their true value because the state
% reflects that they were the weighted contribution to the moving average.
% Hence we need to multiple by the inverse of the weighted contribution.
movingMeanFactor = 10;

additionalStateMean = movingMeanFactor*additionalStateMean;
additionalStateVar = movingMeanFactor*additionalStateVar;
updatedStateMean = (1-factor)*runningStateMean + factor*additionalStateMean;
if factor ==1
    updatedStateVar = additionalStateVar;
else
    updatedStateVar = (1-factor) .* (runningStateVar + runningStateMean.^2) ...
        + factor .* (additionalStateVar + additionalStateMean.^2) ...
        - updatedStateMean.^2;
end
end

function [iteration,valAccur,valLoss] = pollForValidationResponse(validationResponseSignal,executionEnvironment,logFile,Q,maxPollTime)
iteration = [];
valAccur = [];
valLoss = [];
timeout = 1;
if nargin>4
    timeout = maxPollTime;
end

% The signals only facilitate parallel constructs, such as spmd and parfor.
% if (single)gpu mode then a work-around is to
% wrap the signal queries in spmd blocks.
if ~isempty(validationResponseSignal)
    if executionEnvironment=="multi-gpu"
        [data,dataAvailable] = poll(validationResponseSignal,timeout);
    elseif executionEnvironment == "gpu"
        spmd
            if labindex == 1
                [data,dataAvailable] = poll(validationResponseSignal,timeout);
            end
        end
        data = data{1};
        dataAvailable = dataAvailable{1};
    end
    %         fprintf(logFile,"inside pollForValidation dataAvail=%d numel(data)=%d...\r\n",dataAvailable,numel(data));
    if dataAvailable
        iteration = data{3};
        valAccur = data{7};
        valLoss = data{8};
        %         fprintf(logFile,"   iter=%d,acc=%f %f,loss=%f\r\n",iteration,valAccur(1),valAccur(2),valLoss);
        send2Q(Q,executionEnvironment,data);
    end
end
end

function stop = anyStopFlag()
global g_endOfIterationStop g_hardStop g_endOfEpochStop
stop = any([g_endOfIterationStop g_hardStop g_endOfEpochStop]);
end

function pollForStopFlags(stopTrainingSignal,executionEnvironment,maxPollTime)
global g_hardStop g_endOfIterationStop g_endOfEpochStop
try
    timeout = 1;
    if nargin>2
        timeout = maxPollTime;
    end
    % The signals only facilitate parallel constructs, such as spmd and parfor.
    % if (single)gpu mode then a work-around is to
    % wrap the signal queries in spmd blocks.
    if ~isempty(stopTrainingSignal) && timeout >= 0
        if executionEnvironment=="multi-gpu" %each gpu requires to stop its own loop
            [data,dataAvailable] = poll(stopTrainingSignal,timeout);
        elseif executionEnvironment == "gpu"
            %need to put in parallel mode for poll(). since all gpu's will be
            %enabled by spmd, must limit poll() to only first gpu.
            spmd
                if labindex == 1
                    [data,dataAvailable] = poll(stopTrainingSignal,timeout);
                end
            end
            data = data{1};
            dataAvailable = dataAvailable{1};
        end
        
        if dataAvailable && ~anyStopFlag
            if data{1}
                g_hardStop = data{1};
            end
            if data{2}
                g_endOfIterationStop = data{2};
            end
            if data{3}
                g_endOfEpochStop = data{3};
            end
        end
    end
    if exist('stop_training','file')
        g_endOfIterationStop = true;
%         if labindex == 1
%             send2Q(Q,executionEnvironment,[{"stopping training at End-of-Iteration."} {1}]);
%             fprintf(logFile,"stopping training at End-of-Iteration.\r\n");
%         end
    end
    
    g_hardStop = any(gcat(g_hardStop));
    g_endOfIterationStop = any(gcat(g_endOfIterationStop));
    g_endOfEpochStop = any(gcat(g_endOfEpochStop));
catch ME
    if exist('maxPollTime','var') &&  maxPollTime==-1
        fprintf("pollForStopFlags: warning\n");
    end
    warning([ME.message ':' 'pollForStopFlags(): training will be terminated.'])
    g_hardStop = true;
end
    if exist('maxPollTime','var') &&  maxPollTime==-1
        fprintf("pollForStopFlags: end\n");
    end
end


function savePlots(path,executionEnvironment,epoch,iteration,Q)
figFilename = fullfile(pwd,path,['plot_' num2str(epoch) '_' num2str(iteration)]);
if ~exist(fullfile(pwd,path),'dir')
    mkdir(fullfile(pwd,path))
end
send2Q(Q,executionEnvironment,[{"saveFig"} {figFilename}]);
end

function send2Q(Q,executionEnvironment,data)
% The queues only facilitate parallel constructs, such as spmd and parfor.
% if (single)gpu mode then a work-around is to disable the backroundProcess,
% set the displaQ and validationQ as event based DataQueues and
% wrap the queue sends in spmd blocks.
if executionEnvironment == "multi-gpu"
    if labindex == 1
        send(Q,data);
    end
else
    spmd
        if labindex == 1
            send(Q,data);
        end
    end
end
end