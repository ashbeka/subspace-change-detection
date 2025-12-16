function Y = sbsNormalize(X,varargin)
% Y = sbsNormalize(X)
% Normalizing all column-vectors
% input
%  X: column vectors
% output
%  Y: normalized vectors
%
% It was written by Yasuhiro Ohkawa.
%

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

