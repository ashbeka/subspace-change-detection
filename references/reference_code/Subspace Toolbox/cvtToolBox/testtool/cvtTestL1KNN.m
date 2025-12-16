function [er ev predict_l ddd] ...
    = cvtTestL1KNN(train_x, train_l, test_x, test_l)

train_x = train_x(:,:);
test_x = test_x(:,:);
train_l = train_l(:);
test_l = test_l(:);
% [~, ~,train_l] = unique(train_l(:));
% [~,~,test_l] = unique(test_l(:));

ddd = cvtNorm(train_x, test_x, 1);
[~, predict_index] = min(ddd, [], 1);
predict_l = train_l(predict_index);
predict_l = predict_l(:);

% ER
er = mean(predict_l == test_l);

ev = {};
% EER
% ER_MODEL = cvtEER(Dist*10^50, TEST_LABEL, 1000, 2);
% EER = ER_MODEL.EER;
% ev = OrzEval(ddd, test_l', 'D');

end
