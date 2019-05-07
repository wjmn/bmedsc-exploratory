%% Locations and Paths

% MNIST
%renderedGlob = "./data/renderedForTraining/*.mat";
%trainingLabelsPath = "./data/mnistPreprocessed/inTrainingLabels.mat";
%savePath = "./data/models/";

% LANDOLT C
renderedGlob = "./data/renderedForTrainingLandoltc/*.mat";
trainingLabelsPath = "./data/landoltcPreprocessed/labelsTraining.mat";
savePath = "./data/modelsLandoltc/";

%% RUN


load(trainingLabelsPath); %inTrainingLabels

% MNIST
%labels = inTrainingLabels;

datafiles = dir(renderedGlob);

[nFiles, ~] = size(datafiles);

for iFile = 1:nFiles
    
    datafile = datafiles(iFile);
    filePath = datafile.folder + "/" + datafile.name;
    
    load(filePath) % renderedImages variable
    
    [nTraining, nY, nX] = size(renderedImages);
    
    trainingImages = reshape(renderedImages, [nTraining, (nY * nX)]);
    trainingLabels = labels(1:nTraining);
    
    % Reshape to 1D vector
    model = fitcecoc(trainingImages, trainingLabels);

    nameRegexp = regexp(datafile.name, "[\d\w]*", "match");
    nameStripped = nameRegexp{1};
    save(savePath + nameStripped + "_model.mat", "model");
    
    clear("model");
end