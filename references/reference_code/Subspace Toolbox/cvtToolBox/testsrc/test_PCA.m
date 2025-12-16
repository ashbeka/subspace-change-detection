% test PCA
clear;
load('face.mat');

s = size(imgs);
data_x = reshape(imgs, s(1)*s(2), s(3)*s(4));

pca_dim = 30;

[Y, V, D, K] = cvtPCA(data_x, pca_dim);

%%
contribution_rate = 0.90;
[Y, V, D, K] = cvtPCA(data_x, contribution_rate, 0.1);

%%
contribution_rate = 0.95;
[Y, V, D, K] = cvtPCA(data_x, contribution_rate, 0.1);

%%
contribution_rate = 0.99;
[Y, V, D, K] = cvtPCA(data_x, contribution_rate, 0.1);

%%
contribution_rate = 0.999;
[Y, V, D, K] = cvtPCA(data_x, contribution_rate, 0.1);

%%