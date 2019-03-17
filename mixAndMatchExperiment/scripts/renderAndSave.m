% Renders images and saves them for classification and training.

%% Load data/models and Other Paths

% Input Images
load("./data/mnistPreprocessed/inTrainingImages.mat") % inTrainingImages

% Models
load("./data/models/mnistModel.mat") % mnistModel

% Save path
savePath = "./data/renderedForTraining/";

%% PROCESSORS
% Modify here to add processors for the pipeline.
% TODO - More idiomatic way to do this in Matlab?
processors = {};
processors{1} = struct("processor", @processIntensity, "name", "Intensity");
processors{2} = struct("processor", @processEdge, "name", "Edges");
processors{3} = struct("processor", @(i, s) processMnistBraille(i, s, mnistModel), "name", "MnistBraille");
processors{4} = struct("processor", @(i, s) processMnistMimic(i, s, mnistModel), "name", "MnistMimic");

%% RENDERERS
% Modify here to add renderers to the pipeline.

renderers = {};
renderers{1} = struct("renderer", @renderRegular, "name", "Regular");
renderers{2} = struct("renderer", @renderIrregular, "name", "Irregular");

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

% use only the first thousand images to reduce training time
nImages = 1000;

for ip = 1:nProcessors
    
    processorStruct = processors{ip};
    processor = processorStruct.processor;
    processorName = processorStruct.name;
    
    for ir = 1:nRenderers
        
        rendererStruct = renderers{ir};
        renderer = rendererStruct.renderer;
        rendererName = rendererStruct.name;
        
        for scale = scales
            
            % Scale factor
            scaleFactor = scale / inSideLength;
            
            % Assuming square inputs and outputs
            renderedImages = zeros([nImages, renderFactor * inSideLength, renderFactor * inSideLength]);
            
            for ii = 1:nImages
                
            imInput = squeeze(inTrainingImages(ii, :, :));

            processed = processor(imInput, scaleFactor);
            rendered = renderer(processed, 1/scaleFactor * renderFactor);
            renderedImages(ii, :, :) = rendered;
            
            end
            
            save(savePath + "p" + processorName + "_r" + rendererName + "_s" + scale + ".mat", "renderedImages");
            clear('renderedImages');
            
        end
    end
end

