%% File Paths
trainingPath = "./data/mnistOriginal/train.csv";
testPath = "./data/mnistOriginal/test.csv";

%% Reading and Preprocessing Data
trainingFile = csvread(trainingPath, 1, 0);
testFile = csvread(testPath, 1, 0);

% Convert to images

% Training Set

% Training file - first column is label
[nTraining, nColumns] = size(trainingFile);
sideLength = sqrt(nColumns - 1);

% Only take images
trainingLabels = trainingFile(:, 1);
trainingImages = reshape(trainingFile(:, 2:end), [nTraining, sideLength, sideLength]);

% Transpose resulting images
for i = 1:nTraining
    trainingImages(i, :, :) = transpose(squeeze(trainingImages(i, :, :)));
end

clear('trainingFile');

% Test Set
[nTest, nColumns] = size(testFile);
testImages = reshape(testFile, [nTest, sideLength, sideLength]);

for i = 1:nTest
    testImages(i, :, :) = transpose(squeeze(testImages(i, :, :)));
end

clear('testFile')

%% Divide the training images with a 70/30 split
% As we don't have labels for the test set...

nInTraining = round(nTraining * 0.7);

% Input is already in a random order but consider shuffling in the future
inTrainingImages = trainingImages(1:nInTraining, :, :);
inTestImages = trainingImages(nInTraining + 1:end, :, :);
inTrainingLabels = trainingLabels(1:nInTraining, :, :);
inTestLabels = trainingLabels(nInTraining + 1:end, :, :);

%% Saving

inTrainingImagesPath = "./data/mnistPreprocessed/inTrainingImages.mat";
inTrainingLabelsPath = "./data/mnistPreprocessed/inTrainingLabels.mat";
inTestImagesPath = "./data/mnistPreprocessed/inTestImages.mat";
inTestLabelsPath = "./data/mnistPreprocessed/inTestLabels.mat";

testImagesPath = "./data/mnistPreprocessed/testImages.mat";

save(inTrainingImagesPath, 'inTrainingImages');
save(inTrainingLabelsPath, 'inTrainingLabels');
save(inTestImagesPath, 'inTestImages');
save(inTestLabelsPath, 'inTestLabels');

save(testImagesPath, 'testImages');