function [ER predict_l] ...
    = cvtTestMDAkNN(train_x, train_l, test_x, test_l, PCA_DIM_or_CONTRIBUTION_RATE)

train_x = train_x(:,:);
test_x = test_x(:,:);
train_l = train_l(:);
test_l = test_l(:);

[train_x V0 D K] = orzPCA(train_x, PCA_DIM_or_CONTRIBUTION_RATE);
[train_x V1] = cvtMDA(train_x, train_l, PCA_DIM_or_CONTRIBUTION_RATE);

ddd = cvtNorm(train_x, V1'*V0'*test_x, 2);
[~, predict_index] = min(ddd, [], 1);
predict_l = train_l(predict_index);
predict_l = predict_l(:);

% ER
ER = mean(predict_l == test_l);

% ER = cvtTestL2KNN(train_x, train_l, V1'*V0'*test_x, test_l);

% nClass = max(TRAIN_LABEL(:));
% M = cvtClassMean(TRAIN_Y, TRAIN_LABEL);
% 
% TEST_Y = V1'*V0'*TEST_DATA;
% Dist = cvtL2Norm(M, TEST_Y);
% [var PREDICT_LABEL] = min(Dist,[],1);
% PREDICT_LABEL = PREDICT_LABEL(:);
% 
% % ER
% ER = mean(PREDICT_LABEL == TEST_LABEL);
% 



% EER
% ev = OrzEval(Dist, TEST_LABEL', 'D');

end
