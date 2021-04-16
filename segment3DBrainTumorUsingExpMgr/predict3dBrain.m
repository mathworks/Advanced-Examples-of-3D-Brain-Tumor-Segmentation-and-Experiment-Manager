%
% Copyright 2021 The MathWorks, Inc.

function [dataOut] = predict3dBrain(combinedData, net)
origImgSize     = [];
if istable(combinedData)
    I=[];
    T=[];
    for i=1:height(combinedData)
        I = cat(5,I,combinedData{i,1}{1});
        T = cat(5,T,combinedData{i,2}{1});
    end
elseif numel(size(combinedData))==2
    I = cat(5,combinedData{:,1});
    T = cat(5,combinedData{:,2});
else
    origImgSize     = size(combinedData);
    gt = combinedData(:,:,:,end);
    I = combinedData(:,:,:,1:(end-1));
    %using imresize, not imresize3, because do not want to downsample in
    %depth/slice dimension
    T = categorical(gt,1:2,[{'background'},{'tumor'}]);
end

[Y,YScore,YAllScores] = semanticseg(I,net);
if ~isempty(origImgSize)
% %     Yresized = imresize3(Y,origImgSize(1:3),'nearest');
% %     YScoreResized = imresize3(YScore,origImgSize(1:3),'nearest');
%     sizeY = size(Y);
%     offsetI = floor((origImgSize(1:3)-sizeY)/2);
%     I = I(offsetI(1) + (1:sizeY(1)),...
%           offsetI(2) + (1:sizeY(2)),...
%           offsetI(3) + (1:sizeY(3)),1);
%     T = T(offsetI(1) + (1:sizeY(1)),...
%           offsetI(2) + (1:sizeY(2)),...
%           offsetI(3) + (1:sizeY(3)));
    dataOut = [{I},{T},{Y},{YScore}];
else
    dataOut = [{Y},{YScore},{YAllScores}];
end


end

