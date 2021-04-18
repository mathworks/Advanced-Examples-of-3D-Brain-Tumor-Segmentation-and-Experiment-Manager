# Advanced Examples of 3D Brain Tumor Segmentation and Experiment Manager 

This repository uses code based on the product example for "3-D Brain Tumor Segmentation Using Deep Learning", see 
https://www.mathworks.com/help/releases/R2021a/deeplearning/ug/segment-3d-brain-tumor-using-deep-learning.html?s_tid=doc_srchtitle.  The 3-D Brain Segmentation uses the BRaTS dataset that is volumetric representations of the brain with 4 channels or modalities. The examples here were implemented based on my work with a University of Freiburg research team using a head&neck dataset with 7 modalities.  See 1) Bielak, L., Wiedenmann, N., Berlin, A. et al. Convolutional neural networks for head and neck tumor segmentation on 7-channel multiparametric MRI: a leave-one-out analysis. Radiat Oncol 15, 181 (2020). https://doi.org/10.1186/s13014-020-01618-z

The paper was followed by a presentation I gave to the NVIDIA® GTC conference titled "Scale Your Deep Learning Research from Desk to Cloud with MATLAB: Implementing Multiple AI Experiments for Head and Neck Tumor Segmentation"
with emphasis on showing some advanced features implemented with the Experiment Manager.  I am including in this repository the code I presented and a blog that goes into more detail of the work.

ParameterSweepingWithExpMgr:  modified the Brain Segmentation code to demonstrate how the Experiment Manager App can be used to do a Leave-One-Out analysis, as well as, Bayesian Optimization  for hyperparameter determination.

segment3DBrainTumorUsingExpMgr: incorporates a number of advanced features which were implemented with a custom training loop and parallel communication and constructs.  The two major features are implementing a virtual minibatch size by aggregating losses over sub-iterations which emulate multi-gpu's and background tasks running in the background to facilitate validation and plotting.  This code will also run independent of the experiment manager. See the presentation slides for more information on the included features.  

## Setup 
I recommend running the product based 3D Brain Example first from the link above.  And change the conditions to allow for actual training so that you confirm that your computer configuration is correct for doing this.  The full training will take a long time so you might want to make a plan to stop it early by either reducing the data size or number of epochs, etc. or hit the stop button.  In doing this, it will download the BRaTS dataset and pre-process the dataset.

The example code in this repository will do the same thing if the dataset doesn't exist, but it is better to confirm with original example.  Also when running from the experiment manager it does not allow the use of debugging code, so this is more easily done with the original.  Also, it will be easier to seek help from technical support when first trying to get it correct using the product example.

Copy the repository to a folder.

The examples here are advanced, so I recommend becoming familiar with the basics of the experiment manager app by going throught the product documentation and examples.

ParameterSweepingWithExpMgr includes the experiment setups for Leave-One-Out (LOO) and Bayesian Optimization.
1. start experimentManager app, either from command line or Apps tab
2. open the experiment manager project ParameterSweepingWithExpMgr\ParameterSweepingWithExpMgr.prj
3. In the project there are experiments for either BayesOptimization or Leave-One-Out
4. If you have multiple gpu's you can 
	a) run the default parallel cluster or explicitly specify how many gpu's you want to use with parpool command
		For example,  parpool(4) to run 4 gpu's.
	b) select <Use Parallel> button on the app ribbon.
5) select <Run> button on the app ribbon to begin experimentation

segment3DBrainTumorUsingExpMgr includes the setup for the corresponding experiment.
1. start experimentManager app, either from command line or Apps tab
2. open the experiment manager project segment3DBrainTumorUsingExpMgr\segment3DBrainTumorUsintExpMgr.prj
3. In the project there is a single experiment: Experiment1
4. The script internally uses multi-gpu's by default so you cannot select UseParallel.
5) select <Run> button on the app ribbon to begin experimentation

Note the first Hyperparameters, "trialParams", is an array of strings.  This was done to directly associate two corresponding parameters that are converted to integers in the function.  The first of the pair in each string represents the channel to be left out and the second is the trial number to associated with it for output reporting purposes.  The remainder of hyperparameters are fixed.  Also note that it can be run in single 'gpu' mode by changing the executionEnvironment parameter.


For either of experiment wrapper functions, they can be called directly to support debugging.  Just call them with a params structure that matches the Hyperparameters list.  For example: 
params.initialLearnRate = 5e-3;
params.learnRate = .85e-3;
segment3DBrainTumor_LOOWrapper_single(params);

or

params.trialParams = "0 1";
params.maxEpochs = 3;
params.miniBatchSize = 4;
params.executionEnvironment = "multi-gpu"
params.miniBatchMultiplier = 4;
params.secondaryUI_HideRedundantInfo = 0;
segment3DBrainTumorUsingExpMgr(params,[]);

segment3DBrainTumorUsingExpMgr([],[]) will use default hyperparameters.

The custom training function does not facilitate debugging when in multi-gpu mode.



As noted previously,  the experiment manager app doesn't allow debugging.  However you can still run the function wrappers direcly


Additional information about set up

### MathWorks Products (http://www.mathworks.com)

Requires MATLAB release R#### or newer
- R2021a(https://www.mathworks.com/downloads/web_downloads/)
- Image Processing Toolbox(https://www.mathworks.com/products/image.html)
- Computer Vision Toolbox(https://www.mathworks.com/products/computer-vision.html)
- Parallel Computing Toolbox(https://www.mathworks.com/products/parallel-computing.html)
- Deep Learning Toolbox(https://www.mathworks.com/products/deep-learning.html)


## Installation 
Installation instuctions

Before proceeding, ensure that the below products are installed:  
* R2021a(https://www.mathworks.com/downloads/web_downloads/)
* Image Processing Toolbox(https://www.mathworks.com/products/image.html)
* Computer Vision Toolbox(https://www.mathworks.com/products/computer-vision.html)
* Parallel Computing Toolbox(https://www.mathworks.com/products/parallel-computing.html)
* Deep Learning Toolbox(https://www.mathworks.com/products/deep-learning.html)
 
 


## License
The license for https://github.com/mathworks/Advanced-Examples-Of-3D-Brain-Tumor-Segmentation-And-Exp-Mgr is available in the [LICENSE.TXT](LICENSE.TXT) file in this GitHub repository.

## Community Support
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright 2021 The MathWorks, Inc.


