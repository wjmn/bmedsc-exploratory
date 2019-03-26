%% Locations and Paths

%MNIST
%renderedGlob = "./data/renderedForTesting/*.mat";
%testLabelsPath = "./data/mnistPreprocessed/inTestLabels.mat";
%modelsPath = "./data/models/";

%LANDOLTC
renderedGlob = "./data/renderedForTestingLandoltc/*.mat";
testLabelsPath = "./data/landoltcPreprocessed/labelsTest.mat";
modelsPath = "./data/modelsLandoltc/";

savePath = "./data/";

%% RUN

load(testLabelsPath); %inTestLabels

% FOR MNIST ONLY
%labels = inTestLabels;
%clear("inTestLabels");

datafiles = dir(renderedGlob);

[nFiles, ~] = size(datafiles);

results = {};

for iFile = 1:nFiles
    
    % Load rendered images
    datafile = datafiles(iFile);
    filePath = datafile.folder + "/" + datafile.name;
    load(filePath) % renderedImages variable
    
    % Reshape for model
    [nTest, nY, nX] = size(renderedImages);
    renderedFlat = reshape(renderedImages, [nTest, (nY * nX)]);
    
    % Load model
    nameRegexp = regexp(datafile.name, "[\d\w]*", "match");
    nameStripped = nameRegexp{1};
    load(modelsPath + nameStripped + "_model.mat"); % model variable
    
    % Predictions
    predictions = predict(model, renderedFlat);
    
    % Compare
    nCorrect = sum(predictions == labels(1:nTest));
    propCorrect = nCorrect / nTest;
    
    collectRegexp = ['p', '(?<processor>.+)', '_r', '(?<renderer>.+)', '_s', '(?<size>\d+)' ];
    result = regexp(nameStripped, collectRegexp, "names");
    result.accuracy = propCorrect;
    
    results{iFile} = result;
    clear("result");
    clear("model");
    clear("renderedImages");
    clear("predictions");
end

%% Comparison for the MNIST SVM classifiers

%MNIST
%mnistModelPath = "./data/models/mnistModel.mat";
%load(mnistModelPath); %mnistModel variable
%testImagesPath = "./data/mnistPreprocessed/inTestImages.mat";
%load(testImagesPath); % inTestImages variable
%model = mnistModel;

%LANDOLTC
modelPath = "./data/modelsLandoltc/landoltcModel.mat";
load(modelPath);
testImagesPath = "./data/landoltcPreprocessed/imagesTest.mat";
load(testImagesPath);
inTestImages = images;

[~, nY, nX] = size(inTestImages);

% Assume nTest is same for all
testImages = inTestImages(1:nTest, :, :);
testImagesFlat = reshape(testImages, [nTest, (nY * nX)]);

predictions = predict(model, testImagesFlat);

nCorrect = sum(predictions == labels(1:nTest));
propCorrect = nCorrect / nTest;

result.processor = "Original";
result.renderer = "Original";
result.size = "Original";
result.accuracy = propCorrect;

results{iFile + 1} = result;
%% Save

% Flatten

flattened = {};

for i = 1:(nFiles + 1)
    data = results{i};
    flattened{i, 1} = data.processor;
    flattened{i, 2} = data.renderer;
    flattened{i, 3} = data.size;
    flattened{i, 4} = data.accuracy;
end

% Convert to table

columns = {'Processor', 'Renderer', 'Size', 'Accuracy'};
accuracies = cell2table(flattened, 'VariableNames', columns);


%MNIST
%save(savePath + "accuracies.mat", "accuracies", "-v7.3");
%writetable(accuracies, savePath + "accuracies.csv")

% LANDOLTC
save(savePath + "accuraciesLandoltc.mat", "accuracies", "-v7.3");
writetable(accuracies, savePath + "accuraciesLandoltc.csv")