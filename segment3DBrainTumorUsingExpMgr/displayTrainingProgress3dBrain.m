%
% Copyright 2021 The MathWorks, Inc.

function dataAvailable = displayTrainingProgress3dBrain(Q,maxPollTime)
global g_stopBackgroundProcesses g_expMonitor g_clientStopTrainingSignal
persistent  lineLossTrainCell lineLossValidationCell  lineValBkgdAccurCell lineValNormAccurCell
persistent trainingFig axLoss axAccur iterationUit pnlCharts pnlMontage statusUitxt_1 ...
    statusUitxt_2 pnlTruth pnlLastPred pnlSelectPred numValidations validationList...
    uiMontageTruth uiMontageLast uiMontageSelect grpStopButtons ddValidationIterationSelect

if isa(Q,'parallel.pool.PollableDataQueue')
    [data,dataAvailable] = poll(Q,maxPollTime);% if isBusy
else
    dataAvailable = true;
    data = Q;
end

if dataAvailable
    % if the data struction is empty a new UI will be created
    if isempty(data)
        trainingFig = [];
        iterationUit = [];
        pnlCharts = [];
        pnlMontage = [];
        pnlTruth = [];
        pnlLastPred = [];
        pnlSelectPred = [];
        numValidations = [];
        validationList = [];
        uiMontageTruth = [];
        uiMontageLast = [];
        uiMontageSelect = [];
        ddValidationIterationSelect = [];
        
        trainingFig = uifigure('Name',string(timeofday(datetime)));
        trainingFig.Position = [(trainingFig.Position(1:2)-[0 500]) 1420 1000];
        trainingFig.HandleVisibility = 'on';
        
        lastValAccur = [];
        lastValLoss = [];
        
%         g_expMonitor = [];
        
        displayLog = fopen("displayLog.txt","w");
        status = fclose(displayLog);
        
        %setup Stop Button
        grpStopButtons = uibuttongroup(trainingFig,'Title','Stop Buttons','Position',[10 (trainingFig.Position(4)-90) 340 80]);
        stopHardUib = uibutton(grpStopButtons,'Position',[10 10 100 50],'Text','Now', ...
            'ButtonPushedFcn', @(btn,event) stopSelection(btn,event));
        stopEndOfIterationUib = uibutton(grpStopButtons,'Position',[120 10 100 50],'Text','End-of-Iteration', ...
            'ButtonPushedFcn', @(btn,event) stopSelection(btn,event));
        stopEndOfEpochUib = uibutton(grpStopButtons,'Position',[230 10 100 50],'Text','End-of-Epoch', ...
            'ButtonPushedFcn', @(btn,event) stopSelection(btn,event));
        
        %setup iteration table
        iterationUit = uitable('Parent',trainingFig,'Position',[5 (trainingFig.Position(4)-(850+50)) 1410 350]);
        iterationUit.ColumnName = ...
            ["Epoch","Iteration","Elapsed Hours","Mins","Secs","MiniBatchSize","Loss","LR",...
            "Last_Bkgd_Val_Acc","Last_Norm_Val_Acc","Last_Val_Loss","Validation Time (secs)"];
        iterationUit.ColumnFormat = [{'numeric'},{'numeric'},{'numeric'},{'numeric'},{'numeric'},{'numeric'},{'numeric'},{'numeric'},...
            {'numeric'},{'numeric'},{'numeric'},{'numeric'}];
        iterationUit.RowName = 'numbered';
        validationList = [];
        
        
        % setup status text fields
        statusUitxt_1 = uitextarea(trainingFig,"Position", [5 (iterationUit.Position(2)-30) 1410 25]);
        statusUitxt_2 = uitextarea(trainingFig,"Position", [5 (iterationUit.Position(2)-60) 1410 25]);
        
        % setup charts
        pnlCharts = uipanel(trainingFig,"Position",[5 (trainingFig.Position(4)-(450+50+5)) 1410 405]);
        axLoss=uiaxes(pnlCharts,"Position",[5 (pnlCharts.Position(4)-(400)) 400 350]);
        axLoss.YLabel.String="Training(grn)/Validation(blk) Loss";
        axLoss.XLabel.String="Iteration";
        axLoss.YLim=[0 1];
        
        clear("lineLossTrain");
        lineLossTrain = animatedline(axLoss);
        lineLossTrain.Color = 'g';
        lineLossTrainCell = {lineLossTrain};
        
        clear("lineLossValidation");
        lineLossValidation = animatedline(axLoss);
        lineLossValidation.Color = 'black';
        lineLossValidationCell = {lineLossValidation};
        
        axAccur=uiaxes(pnlCharts,"Position",[425 (pnlCharts.Position(4)-(400)) 400 350]);
        axAccur.YLabel.String="Validation Accuracy";
        axAccur.XLabel.String="Iteration";
        axAccur.YLim=[0 100];
        
        clear("lineValBkgdAccur");
        lineValBkgdAccur = animatedline(axAccur);
        lineValBkgdAccur.Color = 'g';
        lineValBkgdAccurCell = {lineValBkgdAccur};
        
        clear("lineValNormAccur");
        lineValNormAccur = animatedline(axAccur);
        lineValNormAccur.Color = 'r';
        lineValNormAccurCell = {lineValNormAccur};
        
        %setup panel to display montage images
        pnlMontage = uipanel(pnlCharts,"Position",[845 (pnlCharts.Position(4)-365) 540 310]);
        pnlMontage.Scrollable = 'on';
        
        
        return;
    end
    
    
    displayLog = fopen("displayLog.txt","a");

    %have to wait until g_expMonitor is assigned in trainDLNetwork()
    if ~isempty(g_expMonitor)
        if isempty(g_expMonitor.Info)
            g_expMonitor.Info = ["Status_1","Status_2","Iteration","Epoch","MiniBatchSize","LearnRate","Last_Val_Iteration","Last_Bkgd_Val_Acc","Last_Norm_Val_Acc","Last_Val_Loss"];
            g_expMonitor.Metrics = ["Training_Loss","Validation_Loss","Bkgd_Val_Acc","Norm_Val_Acc"];
            groupSubPlot(g_expMonitor,"Loss",["Training_Loss","Validation_Loss"]);
            groupSubPlot(g_expMonitor,"Accuracy",["Bkgd_Val_Acc","Norm_Val_Acc"]);
        end
        if g_expMonitor.Stop
            send(g_clientStopTrainingSignal,[{true} {false} {false}]);
        end
    end
    
    %if status message
    if numel(data) >= 1 && isa(data{1},'string')
        statusText = data{1};
            fprintf(displayLog,"%s\r\n",statusText);
        switch statusText
%             case "initialize_expMonitor"
%                 statusUitxt_1.Value = "";
%                 if ~isempty(g_expMonitor)
%                     updateInfo(g_expMonitor,"Status_1","");
%                 end
%                 g_expMonitor = data{2};
                
            case "Stop backgroundProcesses"
                statusUitxt_1.Value = "";
                if ~isempty(g_expMonitor)
                    updateInfo(g_expMonitor,"Status_1","");
                end
                g_stopBackgroundProcesses = true;
                
            case "validation_started"
                statusUitxt_1.Value = "Validation in Progress";
                isValidating = true;
            case "validation_stopped"
                statusUitxt_1.Value = "Validation Completed";
                isValidating = false;
                
            case "figure_name"
                trainingFig.Name = data{2};
                trainingFig.Tag = data{2};
                
            case "saveFig"
                %             if ~isValidating
%                 warning('off');
                savefig(trainingFig,data{2});
%                 warning('on');
                %             end
                
            case "hideRedundantInfo"
        trainingFig.AutoResizeChildren = 'off';
                axLoss.Visible = false;
                axAccur.Visible = false;
                iterationUit.Visible = false;
                statusUitxt_1.Visible = false;
                statusUitxt_2.Visible = false;

                lockMontage = pnlMontage.Position;
                lockCharts = pnlCharts.Position;
                lockStops = grpStopButtons.Position;
                lockTraining = trainingFig.Position;
                
                lockMontage(1) = 5;
                lockCharts(3) = lockMontage(3) + 10;
                pnlCharts.Position = lockCharts;
                drawnow();
                pnlMontage.Position = lockMontage;        
                drawnow();
        
                lockTraining(3) = lockCharts(3) + 10;
                lockTraining(4) = lockTraining(4) - lockCharts(2);
                lockTraining(2) = lockTraining(2) + lockCharts(2);
                trainingFig.Position = lockTraining;
                drawnow();
                pause(1); %seems to help complete drawing, else steps below don't happen
                
                lockCharts(2) = 5;
                lockStops(2) = lockCharts(2)+lockCharts(4)+5; 
                grpStopButtons.Position = lockStops;
                drawnow();
                pnlCharts.Position = lockCharts;
                drawnow();
                pnlMontage.Position = lockMontage;
                drawnow();
        trainingFig.AutoResizeChildren = 'on';
                
            otherwise
                statusIdx = 1;
                if numel(data) >=2 && isnumeric(data{2})
                    statusIdx = data{2};
                end
                switch statusIdx
                    case 1
                        statusUitxt_1.Value = statusText;
                        if ~isempty(g_expMonitor)
                            updateInfo(g_expMonitor,"Status_1",statusText);
                        end
                    case 2
                        statusUitxt_2.Value = statusText;
                        if ~isempty(g_expMonitor)
                            updateInfo(g_expMonitor,"Status_2",statusText);
                        end
                    otherwise
                        statusUitxt_1.Value = statusText;
                        if ~isempty(g_expMonitor)
                            updateInfo(g_expMonitor,"Status_1",statusText);
                        end
                end
        end
        status = fclose(displayLog);
        drawnow;
        return;
    end
    
    %if data record
    iterationTime = data{1};
    epoch = data{2};
    iteration = data{3};
    actualMiniBatchSize = data{4};
    loss = data{5};
    learnRate = data{6};
    
    valAccur = data{7};
    valLoss = data{8};
    if numel(data)>8
        montageSlices = data{9};
        valElapsedTime = data{10};
    end
    
    
    fprintf(displayLog,"%s ep=%d,iter=%d,loss=%f,valLoss=%f\r\n",timeofday(datetime),epoch,iteration,loss,valLoss);
    
    D = duration(0,0,iterationTime,'Format','hh:mm:ss');
    editInx = [];
    if ~isempty(iterationUit.Data)
        editInx = find(iterationUit.Data(:,find(strcmp(iterationUit.ColumnName,"Iteration")))==iteration);
    end
    if isempty(editInx)
        editInx = size(iterationUit.Data,1)+1;
    end
    fprintf(displayLog,"inx=%d\r\n",editInx);
    status = fclose(displayLog);
    if ~isempty(epoch)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Epoch"))) = epoch;
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Elapsed Hours"))) = floor(hours(D));
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Mins"))) = mod(floor(minutes(D)),60);
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Secs"))) = mod(floor(seconds(D)),60);
        drawnow;
        
        if ~isempty(g_expMonitor)
            updateInfo(g_expMonitor,"Epoch",epoch);
        end
    end
    if ~isempty(iteration)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Iteration"))) = iteration;
        drawnow;
        if ~isempty(g_expMonitor)
            updateInfo(g_expMonitor,"Iteration",iteration);
        end
        
    end
    if ~isempty(actualMiniBatchSize)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"MiniBatchSize"))) = actualMiniBatchSize;
        drawnow;
        if ~isempty(g_expMonitor)
            updateInfo(g_expMonitor,"MiniBatchSize",actualMiniBatchSize);
        end
    end
    if ~isempty(loss)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Loss"))) = loss;
        addpoints(lineLossTrainCell{1},iteration,double(loss));
        drawnow;
        if ~isempty(g_expMonitor)
            recordMetrics(g_expMonitor,iteration,"Training_Loss",loss);
        end
    end
    if ~isempty(learnRate)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"LR"))) = learnRate;
        drawnow;
        if ~isempty(g_expMonitor)
            updateInfo(g_expMonitor,"LearnRate",learnRate);
        end
    end
    
    if ~isempty(valAccur)
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Iteration"))) = iteration;
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Last_Bkgd_Val_Acc"))) = valAccur(1);
        if size(valAccur)<2 % in case it was incomplete due to validation process error
            valAccur(2) = 0;
        end
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Last_Norm_Val_Acc"))) = valAccur(2);
        iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Last_Val_Loss"))) = valLoss;
        if exist('valElapsedTime','var') && ~isempty(valElapsedTime)
            iterationUit.Data(editInx,find(strcmp(iterationUit.ColumnName,"Validation Time (secs)"))) = valElapsedTime;
        end
        
        addpoints(lineLossValidationCell{1},iteration,double(valLoss));
        addpoints(lineValBkgdAccurCell{1},iteration,double(100*valAccur(1)));
        addpoints(lineValNormAccurCell{1},iteration,double(100*valAccur(2)));
        drawnow;
        if ~isempty(g_expMonitor)
            updateInfo(g_expMonitor,"Last_Val_Iteration",iteration,...
                "Last_Bkgd_Val_Acc",valAccur(1), ...
                "Last_Norm_Val_Acc",valAccur(2), ...
                "Last_Val_Loss",valLoss);
            recordMetrics(g_expMonitor,iteration,...
                "Bkgd_Val_Acc",valAccur(1), ...
                "Norm_Val_Acc",valAccur(2), ...
                "Validation_Loss",valLoss);
        end
        lastValAccur = valAccur;
        lastValLoss = valLoss;
    end
    
    imDisplayWidth = 160; %assume square
    pnlWidth = imDisplayWidth + 10;
    maxNumImagesToDisplay = 15; %3x3
    if exist('montageSlices','var') && ~isempty(montageSlices)
        im = imread(montageSlices);
        imSize = size(im);
        im = im(1:maxNumImagesToDisplay*imSize(2)/2,:,:);
        imDisplay = imresize(im,[maxNumImagesToDisplay*imDisplayWidth,2*imDisplayWidth],'nearest');
        imDisplay  = addBoundaryOverlay(imDisplay);
    end
    if isempty(pnlTruth)
        pnlTruth = uipanel(pnlMontage,"Position",[5 5 pnlWidth ((maxNumImagesToDisplay+1)*imDisplayWidth+10)],"Title","Truth");
        uiMontageTruth = uiimage(pnlTruth,"Position",[5 0 imDisplayWidth (maxNumImagesToDisplay+1)*imDisplayWidth]);
        uiMontageTruth.HorizontalAlignment = 'center';
        uiMontageTruth.VerticalAlignment = 'top';
        uiMontageTruth.ScaleMethod = 'none';
        
        pnlLastPred = uipanel(pnlMontage,"Position",[2*175 5 pnlWidth ((maxNumImagesToDisplay+1)*imDisplayWidth+10)],"Title",['Iterations ' num2str(iteration)]);
        uiMontageLast = uiimage(pnlLastPred,"Position",[5 0 imDisplayWidth (maxNumImagesToDisplay+1)*imDisplayWidth]);
        uiMontageLast.HorizontalAlignment = 'center';
        uiMontageLast.VerticalAlignment = 'top';
        uiMontageLast.ScaleMethod = 'none';
        
%         pnlSelectPred = uipanel(pnlMontage,"Position",[175 5 pnlWidth [175 5 pnlWidth (maxNumImagesToDisplay+1)*imDisplayWidth],"Title",['Iterations ' num2str(iteration)]);
        pnlSelectPred = uipanel(pnlMontage,"Position",[175 5 pnlWidth ((maxNumImagesToDisplay+1)*imDisplayWidth+10)]);
%         uiMontageSelect = uiimage(pnlSelectPred,"Position",[5 5 imDisplayWidth (maxNumImagesToDisplay+1)*imDisplayWidth]);
        uiMontageSelect = uiimage(pnlSelectPred,"Position",[5 0 imDisplayWidth (maxNumImagesToDisplay+1)*imDisplayWidth]);
        uiMontageSelect.HorizontalAlignment = 'center';
        uiMontageSelect.VerticalAlignment = 'top';
        uiMontageSelect.ScaleMethod = 'none';
        numValidations = 0;
        
        iterationUit.CellSelectionCallback = [{@validationRowSelection},{ddValidationIterationSelect},{uiMontageSelect},{validationList}];

        ddValidationIterationSelect = uidropdown(pnlSelectPred,...
            "Position",[0 ...
                        (pnlSelectPred.Position(4)-20) ...
                        pnlSelectPred.Position(3) 20], ...
            "Items", {}, "ItemsData", []);
        ddValidationIterationSelect.Enable = 'off';
            
        ddValidationIterationSelect.ValueChangedFcn = [{@validationRowSelection},{iterationUit},{uiMontageSelect},{validationList}];
        
        drawnow;
    end
    
    if exist('montageSlices','var') && ~isempty(montageSlices)
        numValidations = numValidations+1;
        pnlLastPred.Title = ['Iterations ' num2str(iteration)];
        uiMontageLast.ImageSource = imDisplay(:,(imDisplayWidth+1):end,:);
        
        if numValidations == 1
            uiMontageTruth.ImageSource = imDisplay(:,1:imDisplayWidth,:);
%             pnlSelectPred.Title = ['Iterations ' num2str(iteration)];
            uiMontageSelect.ImageSource = imDisplay(:,(imDisplayWidth+1):end,:);
        end
        
        validationList = [validationList;{iteration} {montageSlices}];
        iterationUit.CellSelectionCallback = [{@validationRowSelection},{ddValidationIterationSelect},{uiMontageSelect},{validationList}];
        ddValidationIterationSelect.ValueChangedFcn = [{@validationRowSelection},{iterationUit},{uiMontageSelect},{validationList}];
        ddValidationIterationSelect.Items = [ddValidationIterationSelect.Items,sprintf("Iteration %d",iteration)];
        ddValidationIterationSelect.ItemsData = [ddValidationIterationSelect.ItemsData,iteration];
        if ~ddValidationIterationSelect.Enable
            ddValidationIterationSelect.Value = ddValidationIterationSelect.ItemsData(1);
            ddValidationIterationSelect.Enable = 'on';
        end
        drawnow;
    end
    
    if ~isempty(epoch) && ~isempty(iteration)
        pnlCharts.Title = "Epoch: " + epoch + ",Iteration: " + iteration + ", Elapsed: " + string(D);
        axLoss.XLim = [1 100*floor((iteration+100)/100)];
        axAccur.XLim = [1 100*floor((iteration+100)/100)];
        drawnow;
    end
    
end
end

function validationRowSelection(src,event,correpsondingUICtl,uiMontagePanel,validationList)
if ~isempty(validationList)
    if isa(src,'matlab.ui.control.Table')
        iteration = src.Data(event.Indices(1,1),find(strcmp(src.ColumnName,"Iteration")));
        correpsondingUICtl.Value = iteration;
    elseif isa(src,'matlab.ui.control.DropDown')
        iteration = src.Value;
    end
    inx = find(cell2mat(validationList(:,1)) == iteration);
    im = imread(validationList{inx,2});
    imDisplayWidth = 160; %assume square
    maxNumImagesToDisplay = 15;
    imSize = size(im);
    im = im(1:maxNumImagesToDisplay*imSize(2)/2,:,:);
    
    imDisplay = imresize(im,[maxNumImagesToDisplay*imDisplayWidth,2*imDisplayWidth],'nearest');
    imDisplay  = addBoundaryOverlay(imDisplay);

    uiMontagePanel.ImageSource = imDisplay(:,161:end,:);
%     uiMontagePanel.Parent.Title = ['Iterations ' num2str(iteration)];
end
end

function  imOut  = addBoundaryOverlay(im)
    imOut = im;
    boundaryRow = zeros(size(im(1,:,:)),'like',im);
    dtype = class(im);
    if contains(dtype,'int')
        maxval = intmax(dtype);
    else
        maxval = 1;
    end
    for i=10:20:size(boundaryRow,2)
        boundaryRow(:,i+(1:10),:) = maxval;
    end
    imDisplayWidth = size(im,2)/2;
    for i=1:imDisplayWidth:size(im,1)
        imOut(i,:,:) = boundaryRow;
        imOut((i+imDisplayWidth-1),:,:) = boundaryRow;
    end
end

function btnSelection(btn,event)
switch btn.Text
    case "Refresh"
        drawnow;
end
end

function stopSelection(btn,event)
global g_clientStopTrainingSignal
stop_1 = false;
stop_2 = false;
stop_3 = false;
switch btn.Text
    case "Now"
        stop_1 = true;
        btn.Parent.Parent.Children(3).Value="Stop Now buton pushed...";
        
    case "End-of-Iteration"
        stop_2 = true;
        btn.Parent.Parent.Children(3).Value="Stop End-of-Iteration buton pushed...";
        
    case "End-of-Epoch"
        stop_3 = true;
        btn.Parent.Parent.Children(3).Value="Stop End-of-Epoch buton pushed...";
end
drawnow;
grp = btn.Parent;
for i = 1:numel(grp.Children)
    grp.Children(i).Enable = false;
end

send(g_clientStopTrainingSignal,[{stop_1} {stop_2} {stop_3}]);
end

