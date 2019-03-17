function processed = processIntensity(imOriginal,scale)
%PROCESSINTENSITY Processes a colour image to a binary, downsampled
%image based on intensity.
%   processed = PROCESSINTENSITY(imOriginal, scale) takes a 
%   (height, width, 3) image and a float scale (between 0 and 1) and 
%   returns a (height/scale, width/scale) binary image.

dims = size(imOriginal);

% If has colour channelges
if size(dims(2)) == 3
    binarised = imbinarize(rgb2gray(imOriginal));
else
    binarised = imbinarize(imOriginal);
end

processed = imresize(binarised, scale, "nearest");
end

