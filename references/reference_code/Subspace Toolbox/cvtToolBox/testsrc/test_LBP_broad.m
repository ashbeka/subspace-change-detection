% test cvtLBP_broad

% test cvtLBP

% clear;

foo = @(img) cvtLBP_broad(img, 2, 2, 1);

load('face.mat');
[h w num class_num] = size(imgs);
img = imgs(:,:,1,1);
f = foo(img);
bar(f);
% for i = 1:num*class_num
%     f = cvtLBP(imgs(:,:,i));
% end


%% LBP
clear;
foo = @(img) cvtLBP_broad(img, 2, 2, 1);
fdim = 1024;
% function_test = @(train_x, train_l, test_x, test_l) cvtTestMDA(train_x, train_l, test_x, test_l, 100);
function_test = @(train_x, train_l, test_x, test_l) cvtTestSVM(train_x, train_l, test_x, test_l);

score = cvtDataSetTestFace(foo, fdim, function_test);

disp('LBP score');
disp(score);
%%
clear;
% foo = @(img) cvtLBPmini(img,1);
% foo = @(img) cvtLBPmini(img,2);
% foo = @(img) sqrt(cvtLBPmini(img,1));
% foo = @(img) sqrt(cvtLBPmini(img,2));
% fdim = 16;

% foo = @(img) cvtLBP(img);
% foo = @(img) sqrt(cvtLBP(img));
% fdim = 256;

% foo = @(img) cvtLBP_broad(img, 2, 2, 1);
% foo = @(img) cvtLBP_broad(img, 2, 2, 2);
% foo = @(img) cvtLBP_LAC(img, 2, 2, 1);
% foo = @(img) cvtLBP_LAC(img, 2, 2, 2);
% foo = @(img) sqrt(cvtLBP_broad(img, 2, 2, 1));
% foo = @(img) sqrt(cvtLBP_broad(img, 2, 2, 2));
% foo = @(img) sqrt(cvtLBP_LAC(img, 2, 2, 1));
foo = @(img) sqrt(cvtLBP_LAC(img, 2, 2, 2));
fdim = 1024;

% function_test = @(train_x, train_l, test_x, test_l) cvtTestMDA(train_x, train_l, test_x, test_l, 100);
% function_test = @(train_x, train_l, test_x, test_l) cvtTestL1KNN(train_x, train_l, test_x, test_l);
function_test = @(train_x, train_l, test_x, test_l) cvtTestL2KNN(train_x, train_l, test_x, test_l);
% function_test = @(train_x, train_l, test_x, test_l) cvtTestKai2DistKNN(train_x, train_l, test_x, test_l);

score = cvtDataSetTestFace(foo, fdim, function_test);

disp('LBP score');
disp(score);
