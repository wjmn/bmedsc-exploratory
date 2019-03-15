function rendered = renderIrregular(processed, upscale)
%RENDERIRREGULAR Renders a processed binary image on an irregular phosphene
%grid with circular phosphenes.
%   rendered = RENDERIRREGULAR(processed, upscale) takes a (height, width)
%   binary processed image and renders it to a (height*upscale,
%   width*upscale) binary image (upscale above 1) with irregular size and
%   irregular intensity.


[ydim, xdim] = size(processed);

%% Making a base phosphene map 
xs = repmat(1:xdim, [ydim, 1]);
ys = transpose(repmat(1:ydim, [xdim, 1]));

[thetas, rs] = cart2pol(xs, ys);

% Radius for distortion
%rs = rs .^ 1.2;

% Circular mask
%rs(~(rs < (max(max(rs))*0.6))) = 0;

% Positional noise
thetas = thetas + (thetas .* (rand(ydim,xdim) - 0.5) * 0.1);
rs = rs + (rs .* (rand(ydim, xdim) - 0.5) * 0.1);



% Convert polar coordinates to cartesian
[renderX, renderY] = pol2cart(thetas, rs);

shiftedX = 1 + round(upscale * (renderX + abs(min(min(renderX)))));
shiftedY = 1 + round(upscale * (renderY + abs(min(min(renderY)))));

% Output dimension (max)
renderXdim = 1 + max(max(shiftedX));
renderYdim = 1 + max(max(shiftedY));

% Base matrix with output dimensions for rendering
base = zeros(renderYdim, renderXdim);

map = sub2ind([renderYdim, renderXdim], shiftedY, shiftedX);

% Intensity noise
intensityNoise = (rand(ydim, xdim));

base(map) = processed .* intensityNoise;

% Convolution for phosphene blurring
kwidth = floor(upscale / 2);
kernel1d = -kwidth:kwidth;
kernelX = repmat(kernel1d, kwidth * 2 + 1, 1);
kernelY = transpose(kernelX);
kernel = arrayfun(@(x, y) gauss2d(x, y, 0, 0, 0.5*kwidth, 0.5*kwidth), kernelX, kernelY);

% Covolution kernel for phosphene rendering
scaledKernel = kernel / (max(max(kernel)));

rendered = conv2(base, scaledKernel);
end

