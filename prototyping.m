%% Prototyping

%% Clear variables
clear;

%% Setting the webcam
cam = webcam(1);
preview(cam);


%% Downsampling
down_rate = 0.02;
camdims = str2double(split(cam.Resolution, "x"));

dims = round(camdims * down_rate);
xdim = dims(1);
ydim = dims(2);

%% Making a base phosphene map 
xs = repmat(1:xdim, [ydim, 1]) - (xdim / 2);
ys = transpose(repmat(1:ydim, [xdim, 1])) - (ydim / 2);

[thetas, rs] = cart2pol(xs, ys);

% Processing radius
rs = rs .^ 1.2;

% Circular mask
%rs(~(rs < (max(max(rs))*0.6))) = 0;

% Positional noise
thetas = thetas + (thetas .* (rand(ydim,xdim) - 0.5) * 0.1);
rs = rs + (rs .* (rand(ydim, xdim) - 0.5) * 0.1);

% Upscaling for rendering
up_rate = 4;

% Convert polar coordinates to cartesian
[x_render, y_render] = pol2cart(thetas, rs);
x_shifted = 1 + round(up_rate * (x_render + abs(min(min(x_render)))));
y_shifted = 1 + round(up_rate * (y_render + abs(min(min(y_render)))));

% Output dimension (max)
render_size_x = 1 + max(max(x_shifted));
render_size_y = 1 + max(max(y_shifted));

% Phosphene map (with output dimensions)
map = sub2ind([render_size_x, render_size_y], x_shifted, y_shifted);

% Base matrix with output dimensions for rendering
base = zeros(render_size_x, render_size_y);

% Intensity noise
intensity_noise = (rand(ydim, xdim));

% Covolution kernel for phosphene rendering
kernel = [0 1 2 1 0; 1 2 4 2 1; 0 1 2 1 0];
normed_kernel = kernel / max(max(kernel));


%% Setting the video player
videoPlayer = vision.VideoPlayer;

%% Loading pre-trained models

keyModel = "./key_detector.mat";
load(keyModel);

%% Settings

% Add new modes here
modes = {"Intensity", "Edges", "Reading", "Keys"};

[i_mode,tf] = listdlg('ListString', modes);

if tf
    mode_selected = modes(i_mode);
    mode = mode_selected{1,1};
end

if mode == "Intensity"
    process = @(raw) process_intensity(raw);
elseif mode == "Edges"
    process = @(raw) process_edges(raw);
elseif mode == "Reading"
    process = @(raw) process_reading(raw);
elseif mode == "Keys"
    process = @(raw) process_keys(raw, detector);
end



%% Processing loop
while true
   raw = snapshot(cam);
   
   % Process
   processed = process(raw);
   
   %processed = arrayfun(@(o, m) o .* m, flattened, mask);
   base(map) = processed .* intensity_noise;
   
   % Display the image.
   videoPlayer(conv2(transpose(base), normed_kernel));
end


%% Processing Functions

% TODO
function processed = process_reading(raw)
    down_rate = 0.02;
    corrected = imtophat(raw, strel('disk', 15));
    bw = im2bw(corrected);
    ocr_result = ocr(bw);
    downsampled = imresize(bw, down_rate);
    processed = downsampled;
end

function processed = process_intensity(raw)
   % Need this to be global down_rate
   down_rate_1 = 0.2;
   down_rate_2 = 0.1;
   
   downsampled = imresize(raw, down_rate_1 * down_rate_2);
   flattened = im2bw(downsampled);
   processed = flattened;
end

function processed = process_edges(raw)
   % Gobal down_rate
   down_rate_1 = 0.2;
   down_rate_2 = 0.1;
   
   downsampled = imresize(raw, down_rate_1);
   flattened = im2bw(downsampled);   
   detected = edge(flattened, 'canny');
   processed = imresize(detected, down_rate_2);
end

function processed = process_keys(raw, detector)
    down_rate_1 = 0.25;
    down_rate_2 = 0.08;
    
    downsampled = imresize(raw, down_rate_1);
    [bboxes,~] = detect(detector, downsampled);
    if size(bboxes) == [1 4]
        downsampled(:, :, :) = 0;
        downsampled(bboxes(2):bboxes(2) + bboxes(4), bboxes(1):bboxes(1) + bboxes(3), :) = 255;
    end
    resized = imresize(downsampled, down_rate_2);
    flattened = im2bw(resized);
    processed = flattened;
end
