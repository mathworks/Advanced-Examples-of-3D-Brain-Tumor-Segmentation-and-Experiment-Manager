function downloadTrained3DUnetSampleData(net_url,sampledata_url,destination)
% Download the pretrained 3-D U-Net network and sample test volume and
% label data.
%
% Copyright 2019 The MathWorks, Inc.

% Download the 3-D U-Net
filename = 'brainTumor3DUNetValid.mat';
imageDirFullPath = fullfile(destination,'trained3DUNet');
imageFileFullPath = fullfile(imageDirFullPath,filename);

if ~exist(imageFileFullPath,'file')
    fprintf('Downloading pretrained 3-D U-Net for BraTS data set.\n');
    fprintf('This will take several minutes to download...\n');
    if ~exist(imageDirFullPath,'dir')
        mkdir(imageDirFullPath);
    end
    websave(imageFileFullPath,net_url);
    fprintf('Done.\n\n');
else
    fprintf('Pretrained 3-D U-Net model for BraTS data set already exists.\n\n');
end

% Download the sample test data
imageDataLocation = fullfile(destination,'sampleBraTSTestSetValid');
if ~exist(imageDataLocation, 'dir')
    fprintf('Downloading sample BraTS test dataset.\n');
    fprintf('This will take several minutes to download and unzip...\n');
    untar(sampledata_url,destination);
    fprintf('Done.\n\n');
else
    fprintf('Sample BraTS test data set already exists.\n\n');
end