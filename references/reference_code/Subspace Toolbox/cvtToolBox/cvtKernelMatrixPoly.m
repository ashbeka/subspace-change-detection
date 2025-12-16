function [ K ] = cvtKernelMatrixPoly(X,varargin)
% 多項式カーネルのグラム行列を計算する．
% 引数が二つの場合：
% 第一引数をデータＸ，dim x num
% 第二引数を多項式カーネルのパラメータ M として計算
% num x num のマトリックスを返す
%
% 引数が三つの場合：
% 第一引数をデータＸ，dim x num1
% 第二引数をデータＸ，dim x num2
% 第三引数をガウシアンカーネルのパラメータ sigma2 として計算
% num1 x num2 のマトリックスを返す
%

% disp('特徴空間上でノルムの正規化を行っている');
if nargin < 2
   error('error');
end

if nargin == 2
   M = varargin{1};
   K = (X'*X +M(1) ).^M(2);
   dK = sqrt(diag(K));
   for I = 1:size(K,1)
      K(I,:) = K(I,:)/dK(I);
      K(:,I) = K(:,I)/dK(I);
   end
end

if nargin == 3
   Y = varargin{1};
   M = varargin{2};
   clear varargin;
   K = (X'*Y + M(1) ).^M(2);
   dK1 = sqrt((sum(X.^2,1) + M(1)).^M(2));
   dK2 = sqrt((sum(Y.^2,1) + M(1)).^M(2));
   for I1=1:size(K,1)
      K(I1,:) = K(I1,:)/dK1(I1);
   end
   for I2=1:size(K,2)
      K(:,I2) = K(:,I2)/dK2(I2);
   end
end


if nargin > 3
   error('error');
end
