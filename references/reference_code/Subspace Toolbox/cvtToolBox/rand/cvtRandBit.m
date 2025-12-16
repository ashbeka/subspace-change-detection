function y = cvtRandBit(highbit_num, digit_num)

swapindex_a = cvtRandInt([1 digit_num], 1, digit_num);%ceil((digit_num-1)*rand(1,digit_num)+1);
swapindex_b = cvtRandInt([1 digit_num], 1, digit_num);%ceil((digit_num-1)*rand(1,digit_num)+1);

y = [ones(1, highbit_num), zeros(1,(digit_num-highbit_num))];

for i = 1:numel(swapindex_a)
   % swap
   tmp = y(swapindex_a(i));
   y(swapindex_a(i)) = y(swapindex_b(i));
   y(swapindex_b(i)) = tmp;
end

y = logical(y);

end
