%
% Copyright 2021 The MathWorks, Inc.

function [bimlblds] = generateRandomBlockedImageLabelDatastore(filenamePrefix,preprocessDataLoc,volLoc,lblLoc,miniBatchSize,inputBlockSize,outputBlockSize,patchPerImage,numGPU)
volDir = dir(volLoc);
volDir = volDir(3:end);
volDir = volDir(arrayfun(@(x) x.isdir,volDir));
volNames = arrayfun(@(x) [volLoc filesep x.name],volDir,'UniformOutput',false);
bim_s = blockedImage(volNames,'Adapter',images.blocked.MATBlocks,'BlockSize',inputBlockSize);

lblDir = dir(lblLoc);
lblDir = lblDir(3:end);
lblDir = lblDir(arrayfun(@(x) x.isdir,lblDir));
lblNames = arrayfun(@(x) [lblLoc filesep x.name],lblDir,'UniformOutput',false);
blbl_s = blockedImage(lblNames,'Adapter',images.blocked.MATBlocks,'blockSize',outputBlockSize);

tempLoc = fullfile(preprocessDataLoc,'tempTr');

if ~exist([filenamePrefix '_tumorBlocks.mat'],'file')
    if exist(fullfile(tempLoc,[filenamePrefix '_tumorBlocks.mat']),'dir')
        rmdir(fullfile(tempLoc,[filenamePrefix '_tumorBlocks.mat']),'s');
    end
    tumorMask = apply(blbl_s, @(x) x.Data==1,"DisplayWaitbar",false,"OutputLocation",fullfile(tempLoc,[filenamePrefix '_tumorBlocks.mat']));
    blsTumor = selectBlockLocations(bim_s,'Masks',tumorMask,"BlockOffsets",ceil(inputBlockSize/2),'InclusionThreshold',0,'ExcludeIncompleteBlocks',true);
    save([filenamePrefix '_tumorBlocks.mat'],'tumorMask','blsTumor');
else
    load([filenamePrefix '_tumorBlocks.mat']);
end

if ~exist([filenamePrefix '_bkgdBlocks.mat'],'file')
    if exist(fullfile(tempLoc,[filenamePrefix '_bkgdBlocks.mat']),'dir')
        rmdir(fullfile(tempLoc,[filenamePrefix '_bkgdBlocks.mat']),'s');
    end
    bkgdMask = apply(blbl_s, @(x) x.Data==0,"DisplayWaitbar",false,"OutputLocation",fullfile(tempLoc,[filenamePrefix '_bkgdBlocks.mat']));
    blsBkgd = selectBlockLocations(bim_s,'Masks',bkgdMask,"BlockOffsets",ceil(inputBlockSize/2),'InclusionThreshold',.9,'ExcludeIncompleteBlocks',true);
    save([filenamePrefix '_bkgdBlocks.mat'],'bkgdMask','blsBkgd');
else
    load([filenamePrefix '_bkgdBlocks.mat']);
end

ratioBkgd2Tumor = floor(numel(blsBkgd.ImageNumber)/numel(blsTumor.ImageNumber));
imBlsCombined = blockLocationSet(...
    [blsBkgd.ImageNumber;repmat(blsTumor.ImageNumber,ratioBkgd2Tumor,1)],...
    [blsBkgd.BlockOrigin;repmat(blsTumor.BlockOrigin,ratioBkgd2Tumor,1)],...
    blsBkgd.BlockSize,blsBkgd.Levels);
                
im2lblDiff =  round((inputBlockSize-outputBlockSize)/2);
pxBlsCombined = blockLocationSet(...
    imBlsCombined.ImageNumber,...
    imBlsCombined.BlockOrigin(:,1:end-1) + im2lblDiff - 1,...
    outputBlockSize,...
    imBlsCombined.Levels);

bimds = blockedImageDatastore(bim_s,"BlockLocationSet",imBlsCombined,"ReadSize",patchPerImage);
blblds = blockedImageDatastore(blbl_s,"BlockLocationSet",pxBlsCombined,"ReadSize",patchPerImage);
bimlblCombined = combine(bimds,blblds);

bimlblds = randomBlockedImageLabelDatastore(bimlblCombined,["background" "tumor"]);

%need to shuffle here before training partitioning
bimlblds = shuffle(bimlblds);
end


