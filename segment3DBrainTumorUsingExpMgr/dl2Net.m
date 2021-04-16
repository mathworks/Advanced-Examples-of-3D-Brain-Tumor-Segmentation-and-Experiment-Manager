%
% Copyright 2021 The MathWorks, Inc.

function net = dl2Net(dlnet,outputLayers)
lgraph = layerGraph(dlnet);
currentEnd = lgraph.Layers(end);
lgraph = addLayers(lgraph,outputLayers);
lgraph = connectLayers(lgraph,currentEnd.Name,outputLayers(1).Name);
net = assembleNetwork(lgraph);
end