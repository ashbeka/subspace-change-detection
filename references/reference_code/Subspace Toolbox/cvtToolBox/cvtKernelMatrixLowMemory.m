function [ K ] = cvtKernelMatrixLowMemory(X,varargin)
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
    [nDim, nNum] = size(X);
    K = zeros(nNum,nNum);
    for I1=1:nNum
        for I2=1:nNum
            tmpX = X(:,I1)-X(:,I2);
            tmpE = exp(-tmpX'*tmpX/sigma2);
            K(I1,I2) = tmpE;
            K(I2,I1) = tmpE;
        end
    end
end


if nargin > 2
    error('error');
end
