trainingDataPath = "./data/labels/key_labels.mat";

trainingData = objectDetectorTrainingData(gTruth);

inputLayer = imageInputLayer([20, 40, 3]);

filterSize = [3 3];
numFilters = 32;

middleLayers = [
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)   
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)  
    reluLayer() 
    maxPooling2dLayer(3, 'Stride',2)    
    ];

finalLayers = [
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(width(trainingData))
    softmaxLayer()
    classificationLayer()
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];


% Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
% for finer resolution for multiscale object detection.
detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
    'NegativeOverlapRange', [0 0.3], ...
    'PositiveOverlapRange', [0.6 1], ...
    'NumRegionsToSample', [256 128 256 128], ...
    'BoxPyramidScale', 1.2);