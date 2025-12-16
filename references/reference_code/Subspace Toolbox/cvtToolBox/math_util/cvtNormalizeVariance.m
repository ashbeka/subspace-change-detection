function Y = cvtNormalizeVariance(X)
% Y = cvtNormalize(X)
%
% input
%  X: column vectors
% output
%  Y: normalized vectors
%
% It was written by ohkawa on July 7,2009. 
% 
A = var(X(:,:),1,2);
Y = reshape(diag(1./sqrt(A))*X(:,:),size(X));



