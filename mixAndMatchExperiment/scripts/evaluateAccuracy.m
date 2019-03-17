%% Locations and Paths

renderedGlob = "./data/renderedForTesting/*.mat";
testLabelsPath = "./data/mnistPreprocessed/inTestLabels.mat";

modelsPath = "./data/models/";

savePath = "./data/";

%% RUN

load(testLabelsPath); %inTestLabels
labels = inTestLabels;
clear("inTestLabels");

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
end

%% Save

save(savePath + "accuracyResults.mat", "results");