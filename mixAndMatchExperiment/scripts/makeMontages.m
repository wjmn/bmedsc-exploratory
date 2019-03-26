% Makes montages of specific images.

%% Load data/models and Other Paths

% MNIST
% FOR TRAINING SET
%load("./data/mnistPreprocessed/inTrainingImages.mat") % inTrainingImages
%images = inTrainingImages;
%clear("inTrainingImages");
% Models
%load("./data/models/mnistModel.mat") % mnistModel

% LANDOLTC
load("./data/landoltcPreprocessed/imagesTraining.mat");
load("./data/modelsLandoltc/landoltcModel.mat");

savePath = "./graphics/landoltc_";

%% PROCESSORS
% Modify here to add processors for the pipeline.
% TODO - More idiomatic way to do this in Matlab?
processors = {};
processors{1} = struct("processor", @processIntensity, "name", "Intensity");
processors{2} = struct("processor", @processEdge, "name", "Edges");
processors{3} = struct("processor", @(i, s) processLandoltMimic(i, s, model), "name", "LandoltMimic");
%processors{3} = struct("processor", @(i, s) processMnistBraille(i, s, mnistModel), "name", "MnistBraille");
%processors{4} = struct("processor", @(i, s) processMnistMimic(i, s, mnistModel), "name", "MnistMimic");

%% RENDERERS
% Modify here to add renderers to the pipeline.

renderers = {};
renderers{1} = struct("renderer", @renderRegular, "name", "Regular");
renderers{2} = struct("renderer", @renderIrregular, "name", "Irregular");
renderers{3} = struct("renderer", @renderIrregularChanging, "name", "IrregularChanging");

%% SCALES
% Modify here to change scales of renderer phosphene map

% MNIST known to be 28x28
inSideLength = 28;

% Render factor size to be multiplied to inSideLength for the phosphene rendererd image
renderFactor = 2;

% side length in phosphenes, assuming a square map
scales = [3, 5, 10];


%% RUN

[~, nProcessors] = size(processors);
[~, nRenderers] = size(renderers);

imInput = squeeze(images(3, :, :)) / 255;

% scale
imwrite(imInput, savePath + "original.png");

for scale = scales

    % Scale factor
    scaleFactor = scale / inSideLength;
    
    montageCells = {};
    
    for ip = 1:nProcessors

        processorStruct = processors{ip};
        processor = processorStruct.processor;
        processorName = processorStruct.name;
        
        processed = processor(imInput, scaleFactor);
        
        montageCells{end+1} = imresize(processed, (renderFactor * inSideLength / scale), "nearest");

        for ir = 1:nRenderers

            rendererStruct = renderers{ir};
            renderer = rendererStruct.renderer;
            rendererName = rendererStruct.name;

            rendered = renderer(processed, 1/scaleFactor * renderFactor);
            montageCells{end + 1} = rendered;
            
        end
    end
    
    montageImage = montage(montageCells, 'BorderSize', 5, 'BackgroundColor', [0.5 0.5 0.5], 'Size', [nProcessors, nRenderers + 1]);
    montageAxes = getframe(gca);
    imwrite(montageAxes.cdata, savePath + "s" + scale + ".png");
    clear("montageCells");
    clear("montageImage");
    clear("montageAxes");
    
end

