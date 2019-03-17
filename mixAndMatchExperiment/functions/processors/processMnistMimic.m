function processed = processMnistMimic(imOriginal,scale, mnistModel)
%PROCESSMNISTMIMIC Processes a colour image to a binary, downsampled
%image, converting digits to a clean "mimicked" representation using a
%pre-trained multiclass SVM image classifier. This will ONLY work with the
%MNIST dataset and expects a matrix with dimensions [28, 28] and values in
%the range 0 and 1.
%   processed = PROCESSMNISTMIMIC(imOriginal, scale, model) takes a 
%   (height, width, 3) image and a float scale (between 0 and 1) and 
%   returns a (height/scale, width/scale) binary image.

% MUST be 28 x 28!
[nX, nY] = size(imOriginal);

% Reshape the MNIST image to a row vector (as expected for the SVM)
imInput = reshape(imOriginal, [1, nX * nY]);

% Predict
prediction = predict(mnistModel, imInput);


% Choose digit representation if 3x3 or 5x5

% Assume nX == nY
if nX * scale > 3
    digitToBraille = {};
    digitToBraille{'1'} = [0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0];
    digitToBraille{'2'} = [1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1; 1 0 0 0 0; 1 1 1 1 1];
    digitToBraille{'3'} = [1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1];
    digitToBraille{'4'} = [1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1];
    digitToBraille{'5'} = [1 1 1 1 1; 1 0 0 0 0; 1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1];
    digitToBraille{'6'} = [1 1 1 1 1; 1 0 0 0 0; 1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1];
    digitToBraille{'7'} = [1 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1];
    digitToBraille{'8'} = [1 1 1 1 1; 1 0 0 0 1; 1 1 1 1 1; 1 0 0 0 1; 1 1 1 1 1];
    digitToBraille{'9'} = [1 1 1 1 1; 1 0 0 0 1; 1 1 1 1 1; 0 0 0 0 1; 1 1 1 1 1];
    digitToBraille{'0'} = [1 1 1 1 1; 1 0 0 0 1; 1 1 1 1 1; 1 0 0 0 1; 1 1 1 1 1];
else

    % Cell array of digits to braille
    % Can this be moved outside of function for optimisation?
    digitToBraille = {};
    digitToBraille{'1'} = [0 1 0; 0 1 0; 0 1 0];
    digitToBraille{'2'} = [1 1 1; 0 1 0; 1 1 1];
    digitToBraille{'3'} = [1 1 1; 0 1 1; 1 1 1];
    digitToBraille{'4'} = [1 0 1; 1 1 1; 0 0 1];
    digitToBraille{'5'} = [0 1 1; 0 1 1; 1 1 1];
    digitToBraille{'6'} = [1 0 0; 1 1 1; 1 1 1];
    digitToBraille{'7'} = [1 1 1; 0 0 1; 0 0 1];
    digitToBraille{'8'} = [1 1 1; 1 1 1; 1 1 1];
    digitToBraille{'9'} = [1 1 1; 1 1 1; 0 0 1];
    digitToBraille{'0'} = [1 1 1; 1 0 1; 1 1 1];
end

% Rescale
outputXdim = 3;
outputYdim = 3;
imOutput = digitToBraille{num2str(prediction)};

% Upscale back, assuming nX == nY, to a binary image
processed = imresize(imOutput, ((nX * scale) / outputXdim), "nearest");

end

