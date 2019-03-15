%% Clear variables
clear;

%% Setting the webcam
cam = webcam(1);
preview(cam);

filepath = "./data/training/keys/";

for i = 1:200
   raw = snapshot(cam);
   
   downsampled = imresize(raw, 0.25);
   
   imwrite(downsampled, filepath + i + ".jpg");
   
end