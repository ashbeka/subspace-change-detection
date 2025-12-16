% 2d test
% 
% clear;
% load('face.mat');
% 
% %%
% dim = 20;
% X = imgs(:,:,:,2);
% [Y V] = cvt2DPCA(X, dim);
% 
% figure(1);
% imshow(X(:,:,1));
% 
% figure(2);
% vind = 10;
% imagesc(V(:,1:vind)*V(:,1:vind)'*X(:,:,100));
%%
clear;
load('face.mat');
imgs_l = cvtlabel(size(imgs,3), size(imgs,4));
[train_y V1] = cvt2DMDA(imgs(:,:,:), imgs_l);
[train_y V2] = cvt2DMDA(permute(train_y,[2 1 3]), imgs_l);
train_y = permute(train_y, [2 1 3]);
train_y = reshape(train_y, size(train_y,1)*size(train_y,2), size(train_y,3));

load('face2.mat');
test_y = zeros(8, 8, 200*9);
imgs = imgs(:,:,:);
for i = 1:size(test_y,3)
    test_y(:,:,i) = V1' * imgs(:,:,i) * V2;
end
test_y = reshape(test_y, size(test_y,1)*size(test_y,2), size(test_y,3));

cr = cvtTestMEAN(train_y, imgs_l, test_y, imgs_l);

disp(cr);

%%
clear;
load('face.mat');
train_x = reshape(imgs, 40*40, 200*9);
train_l = cvtlabel(size(imgs,3), size(imgs,4));

load('face2.mat');
test_x = reshape(imgs, 40*40, 200*9);
test_l = cvtlabel(size(imgs,3), size(imgs,4));

cr = cvtTestMDA(train_x, train_l, test_x, test_l, 30);
disp(cr);