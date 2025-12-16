function [Y index] = cvtRandSampling(X, num)

% index = cvtRandInt([1 num], 1, size(X, 2));
% Y = X(:, index);

data_num = size(X, 2);
swapindex_a = cvtRandInt([1 data_num], 1, data_num);
swapindex_b = cvtRandInt([1 data_num], 1, data_num);
index = 1:data_num;

for i = 1:data_num
   % swap
   tmp = index(swapindex_a(i));
   index(swapindex_a(i)) = index(swapindex_b(i));
   index(swapindex_b(i)) = tmp;
end

index = index(1:num);
Y = X(:, index);

end
