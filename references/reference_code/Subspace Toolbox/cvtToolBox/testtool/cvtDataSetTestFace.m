function score = cvtDataSetTestFace(function_feature_extractor, feature_dim, function_test)

fdim = feature_dim;
foo = function_feature_extractor;

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

score = function_test(train_x(:,:), train_l, test_x(:,:), test_l);



end