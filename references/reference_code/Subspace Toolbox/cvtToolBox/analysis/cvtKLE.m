function [Y, V, D, K] = cvtKLE(X, sdim )
% [Y, V, D K] = cvtKLE(X, sdim )
% KL展開
% input
%  X: matrix of (m×n), column vectors
%  sdim: 必要な主軸の数
% output
%  Y: matrix of (sdim×n), 各データの主軸に対するパラメータ表示
%  V: matrix of (m×sdim), 主軸
%  D: vector of sdim-dimension, 固有値
%  K: scalar, 累積寄与率
% It was written by ohkawa on July 7,2009. 

X = X(:,:);
N = size(X,2);
R = (X*X'/N);
[V,tmpD]= eigs(R,sdim);
D = diag(tmpD);
Y = V'*X;
K = sum(D)/trace(R);
