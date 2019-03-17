%% Locations and Paths

renderedGlob = "./data/renderedForTraining/*.mat";
trainingLabelsPath = "./data/mnistPreprocessed/inTrainingLabels.mat";

savePath = "./data/models/";

%% RUN

load(trainingLabelsPath); %inTrainingLabels

datafiles = dir(renderedGlob);

[nFiles, ~] = size(datafiles);

for iFile = 1:nFiles
    
    datafile = datafiles(iFile);
    filePath = datafile.folder + "/" + datafile.name;
    

    
    load(filePath) % renderedImages variable
    
    [nTraining, nY, nX] = size(renderedImages);
    
    trainingImages = reshape(renderedImages, [nTraining, (nY * nX)]);
    trainingLabels = inTrainingLabels(1:nTraining);
    
    % Reshape to 1D vector
    model = fitcecoc(trainingImages, trainingLabels);

    nameRegexp = regexp(datafile.name, "[\d\w]*", "match");
    nameStripped = nameRegexp{1};
    save(savePath + nameStripped + "_model.mat", "model");
    
    clear("model");
end