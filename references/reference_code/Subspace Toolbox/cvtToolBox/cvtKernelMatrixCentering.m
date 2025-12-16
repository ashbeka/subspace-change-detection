function [ K ] = cvtKernelMatrixCentering(X, sigma2)
% データをセンタリングしてカーネルマトリックスを計算する．
% 第一引数をデータＸ，dim x num
% 第二引数をガウシアンカーネルのパラメータ sigma2 として計算
% num x num のマトリックスを返す
%
if nargin ~= 2
   error('error');
else
   K = cvtKernelMatrix(X, sigma2);
   ell = size(K,1);
   D = sum(K) / ell;
   E = sum(D) / ell;
   J = ones(ell, 1) * D;
%    K = K - J - J' + E * ones(ell, ell);
end

