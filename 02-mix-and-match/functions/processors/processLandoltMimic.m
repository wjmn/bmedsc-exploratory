function processed = processLandoltMimic(imOriginal,scale, mnistModel)
%PROCESSLANDOLTMIMIC Processes a colour image to a binary, downsampled
%image, converting landolt Cs to a clean representation using a
%pre-trained multiclass SVM image classifier. This will ONLY work with the
%MNIST dataset and expects a matrix with dimensions [28, 28] and values in
%the range 0 and 1.
%   processed = PROCESSMNISTBRAILLE(imOriginal, scale) takes a 
%   (height, width, 3) image and a float scale (between 0 and 1) and 
%   returns a (height/scale, width/scale) binary image.

% MUST be 28 x 28!
[nX, nY] = size(imOriginal);

imInput = reshape(imOriginal, [1, nX * nY]);

% Predict
prediction = predict(mnistModel, imInput);

landoltMimic = {};
landoltMimic{'1'} = [1 0 1; 1 0 1; 1 1 1;];
landoltMimic{'2'} = [1 1 1; 1 0 0; 1 1 1;];
landoltMimic{'3'} = [1 1 1; 1 0 1; 1 0 1;];
landoltMimic{'4'} = [1 1 1; 0 0 1; 1 1 1;];


% Output image as a 2x2 matrix
outputXdim = 3;
outputYdim = 3;
imOutput = landoltMimic{num2str(prediction)};

% Upscale back, assuming nX == nY, to a binary image
processed = imresize(imOutput, ((nX * scale) / outputXdim), "nearest");

end

