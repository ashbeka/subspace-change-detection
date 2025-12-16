load('yaleface_16x16');
s = size(imgs);
data_x = reshape(imgs, s(1)*s(2), s(3));
data_l = imgs_l;

train_x = data_x(:,1:2:end);
train_l = data_l(1:2:end);
test_x = data_x(:,2:2:end);
test_l = data_l(2:2:end);

Random_Forest = Stochastic_Bosque(train_x',train_l, 'ntrees',100,'oobe','y');
[predict_l f_votes]=eval_Stochastic_Bosque(test_x', Random_Forest);
disp(mean(test_l(:) == predict_l(:)));

