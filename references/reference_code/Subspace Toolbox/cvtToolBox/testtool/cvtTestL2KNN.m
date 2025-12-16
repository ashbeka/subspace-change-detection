function [er er_map ev predict_l ddd] ...
    = cvtTestL2KNN(train_x, train_l, test_x, test_l)

train_x = train_x(:,:);
train_l = train_l(:);
test_x = test_x(:,:);
test_l = test_l(:);

num_class = max(train_l(:));
ddd = cvtNorm(train_x, test_x, 2);
[~, predict_index] = min(ddd, [], 1);
predict_l = train_l(predict_index);
predict_l = predict_l(:);

% ER
er = mean(predict_l == test_l);

% ER MAP
count = zeros(num_class, num_class);
for li = 1:numel(predict_l)
   count(test_l(li),predict_l(li)) =  count(test_l(li), predict_l(li)) + 1;
end
er_map = zeros(size(count));
for ci = 1:num_class
   er_map(ci,:) = count(ci,:) / sum(test_l == ci); 
end

% EER
% ER_MODEL = cvtEER(Dist*10^50, TEST_LABEL, 1000, 2);
% EER = ER_MODEL.EER;
% ev = OrzEval(ddd, test_l', 'D');

end
