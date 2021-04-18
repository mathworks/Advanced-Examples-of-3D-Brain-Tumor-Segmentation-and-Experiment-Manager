function [accuracy,loss,montageSlices] = validateModel3dBrain(net,dsValCombined,pixelLabelID,classNames,refresh,displayQ,miniBatchSize)

persistent TScores diceLayer pxdsT T

try
%     if any(contains(properties(dsValCombined),"UnderlyingDatastores"))
%         classNames = dsValCombined.UnderlyingDatastores{2}.ClassNames;
%     else
%         classNames = dsValCombined.ClassNames;
%     end
    if refresh || isempty(TScores)
        if any(contains(properties(dsValCombined),"UnderlyingDatastores"))
            pxdsT = pixelLabelDatastore(dsValCombined.UnderlyingDatastores{2}.Files,classNames,1:numel(classNames),...
                'FileExtensions','.nii','ReadFcn',@niftiread);
        else
            pxdsT = dsValCombined.CombinedDs.UnderlyingDatastores{2};
        end
        cellT = readall(pxdsT);
        T = [];
        for i=1:numel(cellT)
            T=cat(5,T,cellT{i});
        end
        if ~isa(T,'categorical')
            classValues = pixelLabelID;
            if pixelLabelID(1) == 0
                T = T + 1;
                classValues = classValues+1;
            end
%             T = categorical(T,1:numel(classNames),classNames);
            T = categorical(T,classValues,classNames);
        end
%         TScores = single(nnet.internal.cnn.util.dummifyND(T,5));
        TScores = onehotencode(T,4);    
        diceLayer = nnet.internal.cnn.layer.GeneralizedDiceLoss('diceLoss',categorical([]),[],4);
    end
    Y=[];
    YScores = [];
    if any(contains(properties(dsValCombined),"UnderlyingDatastores"))
        numObs = numel(dsValCombined.UnderlyingDatastores{1}.Files);
    else
        numObs = dsValCombined.TotalNumBlocks;
    end
    XminiCombined = preview(dsValCombined);
    if iscell(XminiCombined)
        totalGPURequired = 4*(prod(size(XminiCombined{1})) + prod(size(XminiCombined{2})));
    else
        totalGPURequired = 4*(prod(size(XminiCombined{1,1}{1})) + prod(size(XminiCombined{1,2}{1})));
    end
    obsPerHalfGig = min([numObs,miniBatchSize,floor(5e8/totalGPURequired)]);
catch ME
    rethrow(ME);
end
try
    for i=1:obsPerHalfGig:numObs
        dsXmini = subset(dsValCombined,i:min(i+(obsPerHalfGig-1),numObs));
        XminiCombined = readall(dsXmini);
        
        Ydata = predict3dBrain(XminiCombined,net);
        Ymini = Ydata{1};
        YminiScores = Ydata{3};
        %     [Ymini,~,YminiScores] = semanticseg(Xmini,net,"ExecutionEnvironment","gpu");
        
        Y = cat(4,Y,Ymini);
        YScores = cat(5,YScores,YminiScores);
        maxPollTime = 0;
        %         maxBlockTime = 5;
        %         startTime = tic;
        while true % can't let displayQ buildup %toc(startTime) < maxBlockTime
            if isempty(displayQ) || ~displayTrainingProgress3dBrain(displayQ,maxPollTime) %end if nothing in Q
                break;
            end
        end
        % this condition doesn't work when placed in main training loop in
        % multi-gpu mode, because seems to be blocked by validation thread.
        %  so putting it here.
        assert(~exist('stop_training','file'), ...
            "backgroundProcesses3dBrain:processValidation3dBrain:validateModel3dbrain", ...
            "stop-training detected during validation.");
    end
catch ME
    rethrow(ME);
end
try
    accuracy = predictionAccuracy(Y,T,classNames);
    accuracy = accuracy(1:2);
    loss = diceLayer.forwardLoss(YScores,TScores);
    
    sampleSlices = [];
    montageSlices = [];
    for i=1:numObs
        midSlice = floor(size(T(:,:,:,i),3)/2+1);
        
        sampleSlices = cat(3,T(:,:,midSlice,i),Y(:,:,midSlice,i));
        montageSlices = cat(4,montageSlices,sampleSlices);
    end
catch ME
    rethrow(ME);
end
end

function accuracy = predictionAccuracy(Y,GT,classes)
accuracy = [];
for i = 1:numel(classes)
    GTP = (GT==classes{i});
    YP = (Y==classes{i});
    trueP = sum(GTP(:).*YP(:));
    falseN = sum(GTP(:).*~YP(:));
    accuracy = [accuracy, trueP/(trueP+falseN+eps('single'))];
end
end