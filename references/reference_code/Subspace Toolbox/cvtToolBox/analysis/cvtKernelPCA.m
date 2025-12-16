function [ K, A, C, D, X ] = cvtKernelPCA(X, dim, sigma2)
% カ?[ネルPCA
% K:センタリングしたグラム?s列
% A:
% C:
% D:
% X:射影したデ?[タ

K = cvtKernelMatrixCentering(X,sigma2);

[A, value] = eigs(K, dim, 'LM');

A = A / sqrt(value);

value = diag(value);

C = sum(value) / trace(K);

D = value / size(X,2);

X = A' * K;


end


