% test TestClassifier
clear;

% load('yaleface_64x64.mat');
load('face.mat');
[h w num num_class] = size(imgs);
train_x = reshape(imgs, h*w, num*num_class);
train_l = reshape(cvtLabel(num, num_class), num, num_class);


load('face2.mat');
[h w num num_class] = size(imgs);
test_x = reshape(imgs, h*w, num*num_class);
[dim num num_class] = size(test_x);
test_l = reshape(cvtLabel(num, num_class), num, num_class);

 classifier = @(train_x, train_l, test_x, test_l) cvtTestMDA(train_x, train_l, test_x, test_l, 0.99);

 
 %%
 
%  cvtTestMDA(train_x(:,:), train_l(:), test_x(:,:), test_l(:), 40);
 cvtTestMDA(train_x(:,:), train_l(:), train_x(:,:), train_l(:), 40);
 