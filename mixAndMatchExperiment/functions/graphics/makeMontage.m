function montageImage = makeMontage(original, downscale, upscale, processor, renderer)
%MAKEMONTAGE Makes a "montage" image of the original, processed and
%phosphene-rendered image for comparison.
%   montageImage = MAKEMONTAGE(original, downscale, upscale, processor, renderer) 
%   takes an image (either 2 dimensional, or 3 dimensional with colours), 
%   a processor function, and a renderer function and displays the montage 
%   image of each step. The scales determine how much the image is 
%   downsampled for processing, and subsequently upsampled for rendering.
%   Downscale and upscale should both be above 1.

% To render correctly, make original scaled to between 0 and 1 values
% since the image is a double array for the MNIST dataset
original = original / max(original, [] , 'all');

processed = processor(original, 1 / downscale);
rendered = renderer(processed, upscale);

montageImage = montage({original, processed, rendered}, 'Size', [1, 3]);

end

