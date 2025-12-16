
function [Y,V] = cvtCDA(X, L, alpha)
% 相関判別分析 ver.1.00 by ohkawa
% input
%  X: matrix of (dim×m), column vectors
%  L: vector of m-dimension, L(i) is label of X(:,i).
% output
%  Y: 判別空間へ射影されたX
%  V: 判別軸
%  J: 分離度

disp('sd')
nDim = size(X,1);
label = unique(L);
%クラス数
nClass = size(label,2);
%クラス内共分散行列
Rb= zeros(nDim,nDim);
%クラス間共分散行列
Rw = zeros(nDim,nDim);
%全平均ベクトル

for I=1:nClass;
   tmpX = cvtNormalize( X(:,(L==label(I))));
   Rw = Rw + tmpX*tmpX'/size(tmpX,2);
   mX = cvtNormalize( mean(tmpX,2));
   Rb = Rb + mX*mX';   
end

A = Rb/Rw;
[V,D] = eigs(A,nClass);
Y = V'*X;
