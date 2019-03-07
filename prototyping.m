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
ydim = dims(1);
xdim = dims(2);

%% Conver to polar and create mask
xs = repmat(1:xdim, [ydim, 1]) - (xdim / 2);
ys = transpose(repmat(1:ydim, [xdim, 1])) - (ydim / 2);

thetas = arrayfun(@(x,y) asin(y/x), xs, ys);
rs = arrayfun(@(x,y) sqrt(x^2 + y^2), xs, ys);

mask = transpose((rs < 20));


%% Setting the video player
videoPlayer = vision.VideoPlayer;

%% Processing loop
while true
   raw = snapshot(cam);
   
   % Downsample

   downsampled = imresize(raw, down_rate);
   flattened = im2bw(downsampled);
   
   % Mask
   processed = arrayfun(@(o, m) o .* m, flattened, mask);
   
   % Display the image.
   videoPlayer(processed);
end