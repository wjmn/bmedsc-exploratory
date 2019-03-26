load("./data/landoltcPreprocessed/imagesTraining.mat"); %images
load("./data/landoltcPreprocessed/labelsTraining.mat"); %labels

[nImages, xdim, ydim] = size(images);

% Flatten
images = reshape(images, [nImages, (xdim * ydim)]);

model = fitcecoc(images, labels);

save('./data/modelsLandoltc/landoltcModel.mat', 'model');