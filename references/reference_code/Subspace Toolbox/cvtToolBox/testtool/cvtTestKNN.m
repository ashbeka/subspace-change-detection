function [MAP CR pred_l] = cvtTestKNN(train_x, train_l, test_x, test_l, k)

% train_data.X = train_x;
% train_data.y = train_l;
% model = knnrule(train_data, K);
% pred_l = knnclass(test_x, model);

train_l = train_l(:);
test_l = test_l(:);
pred_l = cvtKNN(test_x, train_x, train_l, k);
pred_l = pred_l(:);
CR = mean(pred_l == test_l);
if(sum(pred_l == -1))
    return;
end
class_num = max(test_l(:));
MAP = zeros(class_num, class_num);
for I = 1:numel(test_l)
    if (pred_l(I) ~= 0)
        MAP(test_l(I), pred_l(I)) =  MAP(test_l(I), pred_l(I)) + 1;
    end
end
for I = 1:class_num
   MAP(I,:) = MAP(I,:) / sum(test_l == I); 
end

end
