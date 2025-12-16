%% LBPLAC
clear;
foo = @(img) cvtVec(imresize(img,[16 16]));
fdim = 16 * 16;
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
[er er_map eer] = cvtTestMDA(train_x(:,:), train_l, test_x(:,:), test_l, 50);
disp(score);


