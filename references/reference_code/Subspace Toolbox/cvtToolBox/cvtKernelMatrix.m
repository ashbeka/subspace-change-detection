function [ K ] = cvtKernelMatrix(X,varargin)
% カーネルマトリックスを計算する．
% 引数が二つの場合：
% 第一引数をデータＸ，dim x num
% 第二引数をガウシアンカーネルのパラメータ sigma2 として計算
% num x num のマトリックスを返す
%
% 引数が三つの場合：
% 第一引数をデータＸ，dim x num1
% 第二引数をデータＸ，dim x num2
% 第三引数をガウシアンカーネルのパラメータ sigma2 として計算
% num1 x num2 のマトリックスを返す
%

if nargin < 2
   error('error');
end

if nargin == 2
   sigma2 = varargin{1};
   Z = repmat(sum(X.^2,1),size(X,2),1);
   D = (2*X'*X-(Z+Z'))/sigma2;
   K=exp(D);
end

if nargin == 3
   Y = varargin{1};
   sigma2 = varargin{2};
   clear varargin;
   ZX = repmat(sum(X.^2,1),size(Y,2),1);
   ZY = repmat(sum(Y.^2,1),size(X,2),1);
   D  = (2*X'*Y - (ZX'+ZY))/sigma2;
   K   = exp(D);
end

if nargin > 3
   error('error');
end
