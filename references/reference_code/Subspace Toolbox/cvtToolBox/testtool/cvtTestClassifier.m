function [er er_map eer] = cvtTestClassifier(data_x, data_l, classifier, k_hold_cross_validation)
% cvtTest
%  == input ==
%   data_x (feature_dim, data_num, class)
%   data_l (data_num, class)
%   classifier [er er_map eer]<-(train_x, train_l, test_x, test_l)
%   k_hold_cross_validation : k-hold-cross-validation paramater
%  == output ==
%   er : error rate
%   er_map : error rate at each classes
%   eer : 
% 

sx = size(data_x);
num_set = sx(2) / k_hold_cross_validation; 

data_x = reshape(data_x, sx(1), k_hold_cross_validation, num_set, sx(3));
sl = size(data_l);
data_l = reshape(data_l, k_hold_cross_validation, num_set, sl(2));

class_num = max(data_l(:));
tmp_er = zeros(num_set, 1);
tmp_er_map = zeros(class_num, class_num, num_set);
tmp_eer = zeros(num_set, 1);

%  cross varidation
for xi = 1:num_set
    train_index = setdiff(1:num_set, xi);
    train_x = data_x(:, :, train_index, :);
    train_l = data_l(:, train_index, :);
    test_x = data_x(:, :, xi, :);
    test_l = data_l(:, xi, :);
    
    train_x = train_x(:,:);
    train_l = train_l(:);
    test_x = test_x(:,:);
    test_l = test_l(:);

    [tmp_er(xi) tmp_er_map(:,:,xi) tmp_eer(xi)] = classifier(train_x, train_l, test_x, test_l);
    
end

er = mean(tmp_er);
er_map = mean(tmp_er_map, 3);
eer = mean(tmp_eer);