function [TEST_CR MAP DDD] = cvtTestMEAN(learn_x, learn_l, test_x, test_l)

learn_l = learn_l(:);
test_l = test_l(:);

learn_m = cvtClassMean(learn_x, learn_l);

DDD = cvtL2Norm(learn_m, test_x);
[~, result_l] = min(DDD,[],1);
TEST_CR = mean(result_l(:) == test_l);
% fprintf('pca cca mda %f\n',TEST_CR);

class_num = max(test_l(:));

MAP = zeros(class_num, class_num);
for I = 1:numel(test_l)
   MAP(test_l(I), result_l(I)) =  MAP(test_l(I), result_l(I)) + 1;
end

for I = 1:class_num
   MAP(I,:) = MAP(I,:) / sum(test_l == I); 
end


end
