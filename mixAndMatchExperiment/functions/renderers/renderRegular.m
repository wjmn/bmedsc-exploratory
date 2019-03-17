function rendered = renderRegular(processed, upscale)
%RENDERREGULAR Renders a processed binary image on a regular phosphene
%grid with circular phosphenes.
%   rendered = RENDERREGULAR(processed, upscale) takes a (height, width)
%   binary processed image and renders it to a (height*upscale,
%   width*upscale) binary image (upscale above 1). 


[ydim, xdim] = size(processed);

%% Making a base phosphene map 
xs = repmat(1:xdim, [ydim, 1]);
ys = transpose(repmat(1:ydim, [xdim, 1]));

renderXdim = floor((xdim - 1) * upscale) + 1;
renderYdim = floor((ydim - 1) * upscale) + 1;
renderXs = floor(((xs - 1) * upscale) + 1);
renderYs = floor(((ys - 1) * upscale) + 1);

% Base matrix with output dimensions for rendering
base = zeros(renderYdim, renderXdim);

map = sub2ind([renderYdim, renderXdim], renderYs, renderXs);

base(map) = processed;

% Convolution for phosphene blurring
kwidth = floor(upscale / 2);
kernel1d = -kwidth:kwidth;
kernelX = repmat(kernel1d, kwidth * 2 + 1, 1);
kernelY = transpose(kernelX);
kernel = arrayfun(@(x, y) gauss2d(x, y, 0, 0, 0.3*kwidth, 0.3*kwidth), kernelX, kernelY);

% Covolution kernel for phosphene rendering
scaledKernel = kernel / (max(max(kernel)));

rendered = conv2(base, scaledKernel);

% Resize to ensure dimensions are predictable
resultYdim = round(ydim * upscale);
resultXdim = round(xdim * upscale);
rendered = imresize(rendered, [resultYdim, resultXdim]);
end

