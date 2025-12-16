%% HOG
clear;

load('face.mat');
[h w num class_num] = size(imgs);


img = imgs(:,:,1,1);

imshow(img);

f = cvtHOG(img);



%% 
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

score = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 50);
disp('HOG score');
disp(score);
