%% Loading Training Data

trainingImagesPath = "./data/mnistPreprocessed/inTrainingImages.mat";
trainingLabelsPath = "./data/mnistPreprocessed/inTrainingLabels.mat";

load(trainingImagesPath); %inTrainingImages
load(trainingLabelsPath); %inTrainingLabels

% Reshpe the training data to a 1 dimensional vector

[nTraining, nX, nY] = size(inTrainingImages);
trainingImages = reshape(inTrainingImages, [nTraining, (nX * nY)]);
trainingLabels = inTrainingLabels;

clear("inTrainingImages");
clear("inTrainingLabels");

%% Svm

% This takes a long long time to run on the full internal training set, so
% I'll only train on the first 10% of the images (2940 images) for now.

proportion = 0.1;
nTake = round(nTraining * proportion);

mnistModel = fitcecoc(trainingImages(1:nTake, :), trainingLabels(1:nTake));

%% Save model

savePath = "./data/models/mnistModel.mat";

save(savePath, "mnistModel");