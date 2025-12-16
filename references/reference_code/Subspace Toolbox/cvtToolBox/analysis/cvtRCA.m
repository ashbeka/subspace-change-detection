function [W, C] = cvtRCA(X, L, alpha)
% X : data
% L : label (chunclet)
% alpha : regularization term (default : 0)
%
% detail :  "Learning Distance functions using Equivalence Relations"


[uni, ~, L] = unique(L);
num_class = numel(uni);
dim = size(X,1);

C = zeros(dim,dim);
for i = 1:num_class
    tmpX = X(:, L == i);
    C = C + size(tmpX,2) * cov(tmpX', 1);
end
    C = C * size(X, 2);
    
if nargin == 3
    C = C + alpha*eye(dim);
end

W = (C)^(-1/2);

% [V D] = eigs(C,150);
% W = V*D^-0.5*V';

W = real(W);

end
