% test cvtLBPLAC
% 
% img = imread('face.jpg');
% imshow(img);
% 
% f = cvtLBPLAC(imresize(rgb2gray(img),[40 40]), 2);
% 
% bar(f);
% 

%%
clear;
load('face.mat');
img = imgs(:,:,1,1);
img = imresize(img, [64 64]);

f = cvtLBPLAC(img, 5,10,8,8);
f = cvtLBPLAC_C(img, 10,10,8,8);


%% 5 78 151 224
for i = 1:900
    f(:,i) = cvtLBPLAC(imgs(:,:,i), 1, 3,5,5);
end

d = cvtNorm(f', f',1);
d = d + 10000*eye(size(d));
imagesc(d);


%%
for i=1:100
    cvtLBPLAC(img, 1,1);
end

for i=1:100
    cvtLBPLAC_C(img, 1,1,5,5);
end


%% LBPLAC
clear;
% foo = @(img) [cvtLBPLAC(img,1); cvtLBPLAC(img,2); cvtLBPLAC(img,3)];
% foo = @(img) [cvtNormalize(cvtLBPLAC(img,1)); cvtNormalize(cvtLBPLAC(img,2)); cvtNormalize(cvtLBPLAC(img,3))];
% foo = @(img) [cvtLBPLAC(img,1); cvtLBPLAC(imresize(img,0.5),1); cvtLBPLAC(imresize(img,0.25),1)];
% foo = @(img) [cvtLBPLAC(img,1,1); cvtLBPLAC(imresize(img,0.5),1,1); cvtLBPLAC(imresize(img,0.25),1,1)];
foo = @(img) [cvtLBPLAC_Complement(img,1,1); cvtLBPLAC_Complement(imresize(img,0.5),1,1); cvtLBPLAC_Complement(imresize(img,0.25),1,1)];
fdim = 1024 * 3;
% fdim = 256 * 3;
load('face.mat');
[h w num class_num] = size(imgs);

train_x = zeros(fdim, num, class_num);
for ci = 1:class_num
for ni = 1:num
    train_x(:,ni,ci) = foo(imgs(:,:,ni,ci));
end
end
train_l = cvtLabel(num, class_num);

load('face2.mat');
test_x = zeros(fdim, num, class_num);
for ci = 1:class_num
for ni = 1:num
    test_x(:,ni,ci) = foo(imgs(:,:,ni,ci));
end
end

test_l = cvtLabel(num, class_num);

%%
score = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 0.999);
disp('LBPLAC score');
disp(score);


