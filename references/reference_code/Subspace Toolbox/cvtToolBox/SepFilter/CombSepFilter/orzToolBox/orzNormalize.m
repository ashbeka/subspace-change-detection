function Y = orzNormalize(X, varargin)
% Y = orzNormalize(X)
% 列ベクトル集合の長さの正規化
% input
%  X: column vectors
% output
%  Y: normalized vectors
%
% It was written by ohkawa on July 7,2009. 
% 
X=double(X);
if nargin < 1
    error('error');
end
if nargin == 1
    D = 2;    
    A = sum(abs(X).^D).^(1/D);
    [s, I] = find(A==0);
    A(I) = 1;
    Y = X./repmat(A,size(X,1),1);
elseif nargin == 2
    D = varargin{1};
    if D==0;
       error('error');
    end
    A = sum(abs(X).^D).^(1/D);
    [s, I] = find(A==0);
    A(I) = 1;
    Y = X./repmat(A,size(X,1),1);
else
    error('error');
end

