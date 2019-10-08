function processed = processEdge(imOriginal,scale)
%PROCESSEDGE Processes a colour image to a binary, downsampled
%image based on edge detection with a Canny detector.
%   processed = PROCESSEDGE(imOriginal, scale) takes a 
%   (height, width, 3) image and a float scale (between 0 and 1) and 
%   returns a (height/scale, width/scale) binary image.


dims = size(imOriginal);

% If has colour channelges
if size(dims(2)) == 3
    binarised = imbinarize(rgb2gray(imOriginal));
else
    binarised = imbinarize(imOriginal);
end
  
% Resize in two steps to try and avoid missing edges
detected = edge(imresize(binarised, scale * 2), 'canny');

processed = imresize(detected, 0.5, "nearest");
end

