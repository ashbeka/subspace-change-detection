% test cvtLBP

% clear;

load('face.mat');
[h w num class_num] = size(imgs);
img = imgs(:,:,1,1);
f = cvtLBP(img);
bar(f);
% for i = 1:num*class_num
%     f = cvtLBP(imgs(:,:,i));
% end


%% LBP
clear;
foo = @(img) [cvtLBP(img)];
fdim = 256;
% function_test = @(train_x, train_l, test_x, test_l) cvtTestMDA(train_x, train_l, test_x, test_l, 100);
function_test = @(train_x, train_l, test_x, test_l) cvtTestSVM(train_x, train_l, test_x, test_l);

score = cvtDataSetTestFace(foo, fdim, function_test);

disp('LBP score');
disp(score);
%%
clear;
foo = @(img) [cvtLBP(img); cvtLBP(imresize(img,0.5)); cvtLBP(imresize(img,0.25))];
fdim = 256*3;
function_test = @(train_x, train_l, test_x, test_l) cvtTestMDA(train_x, train_l, test_x, test_l, 100);

score = cvtDataSetTestFace(foo, fdim, function_test);

disp('LBP score');
disp(score);
