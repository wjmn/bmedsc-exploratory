%% Prototyping

%% Clear variables
clear;

%% Setting the webcam
cam = webcam(1);
preview(cam);


%% Downsampling
down_rate = 0.1;
camdims = str2double(split(cam.Resolution, "x"));

dims = camdims * down_rate;
xdim = dims(1);
ydim = dims(2);

%% Convert to polar and back
xs = repmat(1:xdim, [ydim, 1]) - (xdim / 2);
ys = transpose(repmat(1:ydim, [xdim, 1])) - (ydim / 2);

[thetas, rs] = cart2pol(xs, ys);

up_rate = 3;

[x_render, y_render] = pol2cart(thetas, rs);
x_shifted = 1 + round(up_rate * (x_render + abs(min(min(x_render)))));
y_shifted = 1 + round(up_rate * (y_render + abs(min(min(y_render)))));

render_size = 1 + max(max(x_shifted));
map = sub2ind([render_size, render_size], x_shifted, y_shifted);

base = zeros(render_size, render_size);
%mask = transpose((rs < 20));


%% Setting the video player
videoPlayer = vision.VideoPlayer;

%% Processing loop
while true
   raw = snapshot(cam);
   
   % Downsample

   downsampled = imresize(raw, down_rate);
   flattened = im2bw(downsampled);
   
   % Mask
   %processed = arrayfun(@(o, m) o .* m, flattened, mask);
   base(map) = flattened;
   
   % Display the image.
   videoPlayer(transpose(base));
end