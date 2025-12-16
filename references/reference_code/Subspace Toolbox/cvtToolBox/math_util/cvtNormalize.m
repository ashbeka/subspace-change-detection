function Y = cvtNormalize(X)
% Y = cvtNormalize(X)
% 列ベクトル集合の長さの正規化
% input
%  X: column vectors
% output
%  Y: normalized vectors
%
% It was written by ohkawa on July 7,2009. 
% 
A = sum(X.^2).^0.5;
[s, I] = find(A==0);
A(I) = 1;
Y = X./repmat(A,size(X,1),1);
