%% GLAC test
% 
% img = imread('face.jpg');
% imshow(img);
% 
% f = cvtGLAC(rgb2gray(img));
% 
% bar(f);

%% GLAC descriptor test
img = imread('face.jpg');
imshow(img);

f = cvtGLACDescriptor(rgb2gray(img));

bar(f);

%% GLAC
clear;
glac = @(img) [cvtGLAC(img,1); cvtGLAC(img,2); cvtGLAC(img,3)];
load('face.mat');
[h w num class_num] = size(imgs);

train_x = zeros(256*3, num, class_num);
for ci = 1:class_num
for ni = 1:num
    train_x(:,ni,ci) = glac(imgs(:,:,ni,ci));
end
end
train_l = cvtLabel(num, class_num);


load('face2.mat');
test_x = zeros(256*3, num, class_num);
for ci = 1:class_num
for ni = 1:num
    test_x(:,ni,ci) = glac(imgs(:,:,ni,ci));
end
end

test_l = cvtLabel(num, class_num);

score = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 0.999);
disp('GLAC score');
disp(score);

%% HLAC

clear;
hlac = @(img) [cvtHlac35(img,1); cvtHlac35(img,2); cvtHLAC35(img,3)];

load('face.mat');
[h w num class_num] = size(imgs);


train_x = zeros(35*3, num, class_num);
for ci = 1:class_num
for ni = 1:num
    train_x(:,ni,ci) = hlac(imgs(:,:,ni,ci));
end
end
train_l = cvtLabel(num, class_num);


load('face2.mat');
test_x = zeros(35*3, num, class_num);
for ci = 1:class_num
for ni = 1:num
    test_x(:,ni,ci) = hlac(imgs(:,:,ni,ci));
end
end
test_l = cvtLabel(num, class_num);

score = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 0.999);
disp('HLAC score');
disp(score);

%% HOG
clear;

load('face.mat');
[h w num class_num] = size(imgs);

train_x = zeros(2916, num, class_num);
for ci = 1:class_num
for ni = 1:num
    train_x(:,ni,ci) = cvtVec(cvtHOG(imgs(:,:,ni,ci)));
end
end
train_l = cvtLabel(num, class_num);


load('face2.mat');
test_x = zeros(2916, num, class_num);
for ci = 1:class_num
for ni = 1:num
    test_x(:,ni,ci) = cvtVec(cvtHOG(imgs(:,:,ni,ci)));
end
end
test_l = cvtLabel(num, class_num);

score = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 0,999);
disp('HOG score');
disp(score);

%% veiw base
clear;
load('face.mat');
train_x = reshape(imgs, 40*40, 200*9);
train_l = cvtLabel(size(imgs,3), size(imgs,4));

load('face2.mat');
test_x = reshape(imgs, 40*40, 200*9);
test_l = cvtLabel(size(imgs,3), size(imgs,4));

cr = cvtTestMDA(train_x, train_l, test_x, test_l, 0.999);
disp(cr);
