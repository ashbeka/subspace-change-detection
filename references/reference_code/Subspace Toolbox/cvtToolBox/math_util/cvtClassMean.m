function M = cvtClassMean(X, L)
% ŠeƒNƒ‰ƒX‚Ì•½‹Ï‹‚ß‚é
% input
% X : column vectors
% L : label
% output
% M : Class Mean , DIM x nClass matrix 

% [label,~,L] = unique(L);
% nClass = max(label);
nClass = max(L);
M = zeros(size(X,1), nClass);
for I=1:nClass
        M(:,I) = mean(X(:,L==I), 2);
 end


end
