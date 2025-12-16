function [ K ] = sbsKernelMatrix(X,varargin)
% This function calculates the Kernel Gram Matrix with Gaussian-Kernel.

% When an argument is two:
% The first argument is treated as data X(dim x num1).
% The second argument is treated as parameter of Gaussian-Kernel.
% The (num1 x num1)matrix is returned.
%
% When an argument is three:
% The first argument is treated as data X1(dim x num1).
% The second argument is treated as data X2(dim x num2).
% The third argument is treated as parameter of Gaussian-Kernel.
% The (num1 x num2)matrix is returned.

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
