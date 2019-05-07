imup = imread("up.png");        %1
imright = imread("right.png");  %2
imdown = imread("down.png");    %3
imleft = imread("left.png");    %4
imageChoices = {imup, imright, imdown, imleft};
imageLabels = {1, 2, 3, 4};

nImages = 500;

% Dims are 28 x 28
images = zeros([nImages, 28, 28]);
labels = zeros([nImages 1]);

% FOR LANDOLTC
for i = 1:nImages
    inputNum = randi(4);
    imInput = imageChoices{inputNum};
    label = imageLabels{inputNum};
    images(i, :, :) = imInput;
    labels(i) = label;
end
save("./data/landoltcPreprocessed/imagesTest.mat", "images");
save("./data/landoltcPreprocessed/labelsTest.mat", "labels");