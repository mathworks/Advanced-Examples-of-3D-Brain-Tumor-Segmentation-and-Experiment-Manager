%
% Copyright 2021 The MathWorks, Inc.

classdef randomBlockedImageLabelDatastore < handle & ...
        matlab.mixin.Copyable & ...
        matlab.io.Datastore & ...
        matlab.io.datastore.Partitionable & ...
        matlab.io.datastore.Shuffleable   & ...
        matlab.io.datastore.mixin.Subsettable
    
    properties
        %Source of blocks, an array of blockedImage objects
        CombinedDs (1,1) matlab.io.datastore.CombinedDatastore
%         BatchSize (1,1) {mustBeInteger, mustBePositive} = 128;
        ClassNames (1,:) string
        %ReadSize Number of blocks to read per read() call
        % Default value is 1.
        ReadSize (1,1) double {mustBeInteger} = 1
        
    end
    
    properties (SetAccess = private, Dependent)
        %TotalNumBlocks Total number of blocks available
        TotalNumBlocks(1,1) double {mustBeInteger}
    end
    
    properties (Access = private)
        NextReadIndex(1,1) double = 1
    end
    
    methods
        function obj = randomBlockedImageLabelDatastore(combinedDs,classNames);%,batchSize)
            obj.CombinedDs = combinedDs;
%             obj.BatchSize = min(obj.TotalNumBlocks, batchSize);
            obj.ReadSize = combinedDs.UnderlyingDatastores{1}.ReadSize;
            obj.ClassNames = classNames;
        end
        
        function tf = hasdata(obj)
%             tf = obj.NextReadIndex <= obj.BatchSize ;
            tf = obj.NextReadIndex <= obj.TotalNumBlocks ;
        end
            
        function reset(obj)
            obj.NextReadIndex = 1;
            reset(obj.CombinedDs);
        end
        
        function newds = shuffle(obj)
            % shuffle  Permute order blocks will be read.
            %     NEWDS = shuffle(BIMDS) randomly reorders the read order
            %     of the blocks in BIMDS and returns a new
            %     blockedImageDatastore NEWDS. The original datastore is
            %     unchanged.
            %
            %     Example
            %     -------
            %     bim = blockedImage('tumor_091R.tif');
            %     bimds = blockedImageDatastore(bim);
            %     while hasdata(bimds)
            %         [data, info] = read(bimds);
            %         disp(info);
            %     end
            %
            %     sbimds = shuffle(bimds);
            %     disp('Shuffled Order');
            %     while hasdata(sbimds)
            %         [data, info] = read(sbimds);
            %         disp(info);
            %     end
            %
            %     See also blockedImageDatastore, partition
            
            newds = copy(obj);
            randPermInd = randperm(obj.TotalNumBlocks);

            newds.CombinedDs = subset(obj.CombinedDs,randPermInd);
            newds.reset()
        end
        function [blocks, blockInfo] = read(obj)
            %read Read data and information about the extracted data.
            %  BCELL = read(BIMDS) Returns the data extracted from the
            %  blockedImageDatastore, BIMDS. BCELL is a cell array of block
            %  data of length ReadSize.
            %
            %  [B, INFO] = read(BIMDS) also returns information about where
            %  the data was extracted from the blockedImageDatastore. INFO
            %  is a scalar struct with the following fields. These fields
            %  are arrays if ReadSize>1.
            %
            %    Level        - The level from which this data was read.
            %    ImageNumber  - An index into the bimds.Images array
            %                   corresponding to the blockedImage from
            %                   which this block was read.
            %     Start       - Array subscripts of the first element in
            %                   the block. If BorderSize is specified, this
            %                   subscript can be out-of-bounds for edge
            %                   blocks.
            %     End         - Array subscripts of the last element in the
            %                   block. If BorderSize is specified, this
            %                   subscript can be out-of-bounds for edge
            %                   blocks.
            %     Blocksub    - The block subscripts of the current block.
            %     BorderSize  - The value of the BorderSize parameter.
            %     BlockSize   - The value of the BlockSize parameter.
            %
            %  Example
            %  -------
            %   bim = blockedImage('tumor_091R.tif');
            %   bimds = blockedImageDatastore(bim);
            %   while hasdata(bimds)
            %     [data, info] = read(bimds);
            %     disp(info);
            %   end
            %
            %  See also blockedImageDatastore, readsize
            
            [combinedBlocks,blockInfo] = read(obj.CombinedDs);
            toCategorical=cellfun(@(x) categorical(x+1,1:numel(obj.ClassNames),obj.ClassNames),...
                combinedBlocks(:,2),'UniformOutput',false);
            combinedBlocks(:,2) = toCategorical;
            blocks = cell2table(combinedBlocks,'VariableNames',{'InputImage' 'ResponseImage'});
            obj.NextReadIndex = obj.NextReadIndex+height(blocks);
%             obj.NextReadIndex = min(obj.NextReadIndex,obj.BatchSize);
        end
        
        function pds = partition(obj, numP, idx)
            % partition Return a partitioned part of the blockedImageDatastore.
            %     SUBBIMDS = partition(BIMDS,N,INDEX) partitions DS into N
            %     parts and returns the partitioned Datastore, SUBDS,
            %     corresponding to INDEX. An estimate for a reasonable
            %     value for N can be obtained by using the NUMPARTITIONS
            %     function.
            %
            %     Example
            %     -------
            %     bim = blockedImage('tumor_091R.tif');
            %     bimds = blockedImageDatastore(bim);
            %
            %     bimdsp1 = partition(bimds, 2, 1);
            %     disp('Partition 1');
            %     while hasdata(bimdsp1)
            %         [data, info] = read(bimdsp1);
            %         disp(info);
            %     end
            %
            %     bimdsp2 = partition(bimds, 2, 2);
            %     disp('Partition 2');
            %     while hasdata(bimdsp2)
            %         [data, info] = read(bimdsp2);
            %         disp(info);
            %     end
            %
            % See also blockedImageDatastore, numpartitions, maxpartitions
            
            arguments
                obj (1,1) randomBlockedImageLabelDatastore
                numP (1,1) double {mustBeInteger}
                idx (1,1) double {mustBeInteger}
            end
            pds = copy(obj);
            pds.CombinedDs = partition(obj.CombinedDs,numP,idx);
%             pds.BatchSize = floor(pds.BatchSize/numP);
            pds.reset()
        end
        
        function blocks = preview(obj)
            %  preview   Reads the first block
            %     b = preview(bimds) Returns the first block from the start
            %     of the datastore.
            %
            %     See also blockedImageDatastore, read, hasdata, reset, readall,
            %     progress.
            combinedBlocks = preview(obj.CombinedDs);
            toCategorical=cellfun(@(x) categorical(x+1,1:numel(obj.ClassNames),obj.ClassNames),...
                combinedBlocks(:,2),'UniformOutput',false);
            combinedBlocks(:,2) = toCategorical;
            blocks = cell2table(combinedBlocks,'VariableNames',{'InputImage' 'ResponseImage'});
        end
        
        function amount = progress(obj)
            %  progress  Amount of datastore read.
            %     R = progress(BIMDS) gives the ratio of the datastore
            %     BIMDS that has been read.
            
%             amount = (obj.NextReadIndex - 1) / obj.BatchSize;
            amount = (obj.NextReadIndex - 1) / obj.TotalNumBlocks;
        end
        
        function subds = subset(obj, indices)
            % subset  Returns a new datastore with the specified
            % block indices
            %
            %  SUBDS = subset(BIMDS, INDICES) creates a deep copy of the
            %  input datastore DS containing blocks corresponding to
            %  INDICES. INDICES must be a vector of positive and unique
            %  integer numeric values. INDICES can be a 0-by-1 empty array
            %  and does not need to be provided in any sorted order when
            %  nonempty. The output datastore SUBDS, contains the blocks
            %  corresponding to INDICES and in the same order as INDICES.
            %  INDICES can also be specified as a N-by-1 vector of logical
            %  values, where N is the total number of blocks in the
            %  datastore (TotalNumBlocks property).
            %
            % See also blockedImageDatastore, partition, shuffle
            
%             import matlab.io.datastore.internal.validators.validateSubsetIndices;
%             indices = validateSubsetIndices(indices, obj.TotalNumBlocks, mfilename);
%             
            subds = copy(obj);
            subds.CombinedDs = subset(obj.CombinedDs,indices);
            subds.ClassNames = obj.ClassNames;
            subds.ReadSize = obj.ReadSize;
%             subds.BatchSize = min(obj.BatchSize,numel(indices));
            subds.reset();
        end
        
        function totalNumBlocks = get.TotalNumBlocks(obj)
            totalNumBlocks = obj.CombinedDs.UnderlyingDatastores{1}.TotalNumBlocks;
        end
        
%         function tbl = countEachLabel(obj,params)
%             %countEachLabel Counts the number of pixel labels for each class.
%             %
%             % tbl = countEachLabel(BIMDS) counts the occurrence of each
%             % pixel label for all blocks represented by bimds. Only blocks
%             % that are output by read are used for counting. The output tbl
%             % is a table with the following variable names:
%             %
%             %   Name            - The pixel label class name.
%             %
%             %   PixelCount      - The number of pixels of a given class in 
%             %                     all blocks.
%             %
%             %   BlockPixelCount - The total number of pixels in blocks that
%             %                     had an instance of the given class.
%             %
%             % Note: If BIMDS yields categorical data, the categories of
%             % this data are used as the class names. If BIMDS yields
%             % numeric data, 'Classes' and 'PixelLabelsIDs' must be
%             % provided.
%             %
%             %   [___] = countEachLabel(___, Name, Value) specifies
%             %   additional parameters. Supported parameters include:
%             %
%             %   'UseParallel'         A logical scalar specifying if a new
%             %                         or existing parallel pool should be
%             %                         used. If no parallel pool is active,
%             %                         a new pool is opened based on the
%             %                         default parallel settings. This
%             %                         syntax requires Parallel Computing
%             %                         Toolbox. Default value is false.
%             %    'Classes'            Class names. A cell array of strings
%             %                         or char vectors with the label names.
%             %    'PixelLabelIDs'      Values for each label. A numeric
%             %                         array of values with the same length
%             %                         as 'Classes'. This provides the
%             %                         mapping from numeric values to the
%             %                         label class. Specify RGB values as a
%             %                         Nx3 numeric array.
%             %
%             % Class Balancing
%             % ---------------
%             % The output of countEachLabel, tbl can be used to calculate
%             % class weights for class balancing, for example:
%             %
%             %   * Uniform class balancing weights each class such that each
%             %     has a uniform prior probability:
%             %
%             %        numClasses = height(tbl)
%             %        prior = 1/numClasses;
%             %        classWeights = prior ./ tbl.PixelCount
%             %
%             %   * Inverse frequency balancing weights each class such that
%             %     underrepresented classes are given higher weight:
%             %
%             %        totalNumberOfPixels = sum(tbl.PixelCount)
%             %        frequency = tbl.PixelCount / totalNumberOfPixels;
%             %        classWeights = 1./frequency
%             %
%             %   * Median frequency balancing weights each class using the
%             %     median frequency. The weight for each class c is defined
%             %     as median(imageFreq)/imageBlockFreq(c) where
%             %     imageBlockFreq(c) is the number of pixels of a given
%             %     class divided by the total number of pixels in image
%             %     blocks that had an instance of the given class c.
%             %
%             %        imageBlockFreq = tbl.PixelCount ./ tbl.BlockPixelCount
%             %        classWeights = median(imageBlockFreq) ./ imageBlockFreq
%             %
%             % The calculated class weights can be passed to the
%             % pixelClassificationLayer. See example below.
%             %
%             % Example 1
%             % ---------
%             %   % Count pixel labels from numeric data
%             %   lbim = blockedImage('yellowlily-segmented.png', 'BlockSize', [512 512]);
%             %   lbimds = blockedImageDatastore(lbim);
%             %   countEachLabel(lbimds, ...
%             %     "Classes", ["Background", "Flower", "Leaf", "Background"],...
%             %     "PixelLabelIDs", [0, 1, 2, 3])
%             %
%             % Example 2
%             % ---------
%             %   % Counts pixel labels in a labeled image and calculate
%             %   % class weights for class balancing
%             %
%             %   % Load labeled data
%             %   load('buildingPixelLabeled.mat');
%             %
%             %   % Count pixel labels occurrences in the labeled images
%             %   bimLabelled = blockedImage(label);
%             %
%             %   % Set the block size of the images           
%             %   blockSize = [200 150];
%             %
%             %   % Create a datastore from the image dataset
%             %   blabelds = blockedImageDatastore(bimLabelled, 'BlockSize', blockSize);
%             %
%             %   % Look at the pixel label occurrences of each class.
%             %   tbl = countEachLabel(blabelds);
%             %
%             %   % Class balancing using uniform prior weighting.
%             %   prior = 1/numel(classNames);
%             %   uniformClassWeights = prior ./ tbl.PixelCount
%             %
%             %   % Class balancing using inverse frequency weighting.
%             %   totalNumberOfPixels = sum(tbl.PixelCount);
%             %   frequency = tbl.PixelCount / totalNumberOfPixels;
%             %   invFreqClassWeights = 1./frequency
%             %
%             %   % Class balancing using median frequency weighting.
%             %   freq = tbl.PixelCount ./ tbl.BlockPixelCount
%             %   medFreqClassWeights = median(freq) ./ freq
%             %
%             %   % Pass the class weights to the pixel classification layer.
%             %   layer = pixelClassificationLayer('ClassNames', tbl.Name, ...
%             %       'ClassWeights', medFreqClassWeights)
%             %
%             %
%             % See also pixelClassificationLayer, blockedImage,
%             % blockedImageDatastore
%             
%             arguments
%                 obj (1,1) blockedImageDatastore
%                 params.UseParallel (1,1) logical = false
%                 params.Classes (:,1) string
%                 params.PixelLabelIDs (:,:) double {mustBeNumericOrLogical}
%             end
%             
%             % TODO - ensure BIMDS.Images all have the same underlying type
%             % during construction.For categorical, ensure to check all have
%             % the same classes
%             
%             if obj.Images(1).ClassUnderlying == "categorical"
%                 if isfield(params,'Classes') || isfield(params,'PixelLabelIDs')
%                     assert(false, "Datastore contains categorical blocks. Classes and PixelLabelIDs are not supported");
%                 end
%                 params.Classes = categories(obj.Images(1).InitialValue);
%                 params.HasCats = true;
%             else
%                 %TODO PixelLabelIds must be unique
%                 % TODO - verify ClassUnderlying is numeric/logical
%                 if ~(isfield(params,'Classes') && isfield(params,'PixelLabelIDs'))
%                     assert(false, "Datastore contains numeric blocks. Both Classes and PixelLabelIDs must be specified.");
%                 end
%                 params.HasCats = false;
%                 
%                 % 1xN, Nx1 or Nx3
%                 if size(params.PixelLabelIDs,2)==3 % RGB
%                     % TODO - assert all bims are 2D.
%                     % Change to cell array of 1x1x3's.
%                     params.PixelLabelIDs = reshape(params.PixelLabelIDs,...
%                         [size(params.PixelLabelIDs,1), 1, 3]);
%                     params.PixelLabelIDs = mat2cell(params.PixelLabelIDs,...
%                         ones(1, size(params.PixelLabelIDs,1)), 1, 3);
%                 else
%                     %TODO Assert 1xN or Nx1
%                     params.PixelLabelIDs = params.PixelLabelIDs(:);
%                 end
%             end
%             
%             numClasses = numel(params.Classes);
%             % Make a copy so we do not dirty the state.
%             newds = copy(obj);
%             newds.reset();
%                                     
%             
%             if params.UseParallel
%                 % Set up parallel pool.
%                 p = gcp;
%                 if isempty(p)
%                     error(message('images:bigimageDatastore:couldNotOpenPool'))
%                 end
%                 numPartitions = p.NumWorkers;
%                 
%                 counts          = zeros(numClasses, numPartitions);
%                 blockPixelCount = zeros(numClasses, numPartitions);
%                 
%                 parfor pIdx = 1:numPartitions
%                     subds = partition(newds,numPartitions,pIdx);
%                     [countsSlice, blockPixelCountSlice] = calculateCountsAndPixelCounts(subds, params);
%                     counts(:, pIdx) = countsSlice
%                     blockPixelCount(:,pIdx) = blockPixelCountSlice;
%                 end
%                 
%                 % Aggregate the results of classes in each block from all partitions
%                 blockPixelCount = sum(blockPixelCount,2);
%                 counts = sum(counts,2);
%                 
%             else
%                 [counts, blockPixelCount] = calculateCountsAndPixelCounts(newds, params);
%             end
%             
%             % Combine duplicate classes if any
%             [uniqueClasses, ~, indc] = unique(params.Classes, 'stable');
%             if numel(uniqueClasses) ~= numel(params.Classes)
%                 uniqueCounts = zeros(numel(uniqueClasses),1);
%                 uniqueBlockPixelCount = zeros(numel(uniqueClasses),1);
%                 for cInd = 1:numel(uniqueClasses)
%                     uniqueCounts(cInd) = sum(counts(indc==cInd));
%                     uniqueBlockPixelCount(cInd) = sum(blockPixelCount(indc==cInd));
%                 end
%             else
%                 uniqueCounts = counts;
%                 uniqueBlockPixelCount = blockPixelCount;
%             end
%             
%             tbl = table();
%             tbl.Name            = string(uniqueClasses);
%             tbl.PixelCount      = uniqueCounts;
%             tbl.BlockPixelCount = uniqueBlockPixelCount;
%         end
        
    end
    
    methods (Access = protected)
        function num = maxpartitions(obj)
            num = obj.TotalNumBlocks;
        end
    end

    methods (Hidden)
        
        function n = numobservations(obj)
            n = obj.TotalNumBlocks;
        end
    end
    
 
% function validatePadMethod(val)
% % TODO - tighten this, we use padarray_algo internally which does not do
% % validation.
% if ~isnumeric(val) && ~iscategorical(val) && ~isstruct(val) && ~islogical(val)
%     validatestring(val, {'replicate','symmetric'},mfilename,'PadMethod');
% end
% 
% end
% 
% function mustMatchDims(bs, bims)
% dims = bims(1).NumDimensions;
% assert(numel(bs)==dims,"BorderSize must have same number of elements as dimensions");
% end
% 
% function [counts, blockPixelCount] = calculateCountsAndPixelCounts(ds, params)
% % calculateCountsAndPixelCounts Calculate the counts and blockPixelCounts
% % for all classes in the datastore, ds. 
% 
% numClasses = numel(params.Classes);
% 
% counts = zeros(numClasses,1);
% blockPixelCount = zeros(numClasses,1);
% 
% while hasdata(ds)
%     data = read(ds);
%     for blockIdx = 1:numel(data)        
%         blockSize = size(data{blockIdx});
%         numObs = prod(blockSize);
%         
%         if params.HasCats
%             % TODO - each image should have the same order when categories
%             % is called on it. Else we should sort here. 
%             countsForOneBlock = countcats(data{blockIdx}(:));
%         elseif iscell(params.PixelLabelIDs) % 2D RGB            
%             countsForOneBlock = arrayfun(...
%                 @(x) nnz(all(data{blockIdx} == x{1},3)),...
%                 params.PixelLabelIDs);
%             % Ignore channel dimension
%             numObs = prod(blockSize(1:2));
%         else % numeric
%             countsForOneBlock = arrayfun(...
%                 @(x) nnz(data{blockIdx} == x),...
%                 params.PixelLabelIDs);
%         end        
%         
%         counts = counts + countsForOneBlock;
%         
%         classIdx = countsForOneBlock > 0;        
%         blockPixelCount(classIdx) = blockPixelCount(classIdx) + numObs;
%     end
% end
end
