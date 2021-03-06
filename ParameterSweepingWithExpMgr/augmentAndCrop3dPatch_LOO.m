function patchOut = augmentAndCrop3dPatch_LOO(patchIn,flag,channelToExclude)

% Augment training data by randomly rotating and reflecting image patches.
% Do not augment validation data. For both training and validation data,
% crop the response to the network's output size. Return the image patches
% in a two-column table as required by the trainNetwork function for
% single-input networks.
%
% Copyright 2019 The MathWorks, Inc.
isValidationData = strcmp(flag,'validation');

inpVol = cell(size(patchIn,1),1);
inpResponse = cell(size(patchIn,1),1);

% 5 augmentations: nil,rot90,fliplr,flipud,rot90(fliplr)
fliprot = @(x) rot90(fliplr(x));
augType = {@rot90,@fliplr,@flipud,fliprot};
for id=1:size(patchIn,1) 
    rndIdx = randi(8,1);
    tmpImg =  patchIn.InputImage{id};
    tmpResp = patchIn.ResponsePixelLabelImage{id};
    if rndIdx > 4 || isValidationData
        out =  tmpImg;
        respOut = tmpResp;
    else
        out =  augType{rndIdx}(tmpImg);
        respOut = augType{rndIdx}(tmpResp);
    end
    % can effectively exclude a channel by setting it to a constant
    if ~isempty(channelToExclude) && channelToExclude > 0
        out(:,:,:,channelToExclude) = 0;
    end
    % Crop the response to to the network's output.
    respFinal=respOut(45:end-44,45:end-44,45:end-44,:);
    inpVol{id,1}= out;
    inpResponse{id,1}=respFinal;
end
patchOut = table(inpVol,inpResponse);