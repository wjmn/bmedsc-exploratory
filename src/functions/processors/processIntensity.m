function processed = processIntensity(imOriginal,scale)
%PROCESSINTENSITY Processes a colour image to a binary, downsampled
%image based on intensity.
%   processed = PROCESSINTENSITY(imOriginal, scale) takes a 
%   (height, width, 3) image and a float scale (between 0 and 1) and 
%   returns a (height/scale, width/scale) binary image.

binarised = imbinarize(rgb2gray(imOriginal));
processed = imresize(binarised, scale);
end

