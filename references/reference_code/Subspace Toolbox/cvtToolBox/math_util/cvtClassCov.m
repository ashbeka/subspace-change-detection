function C = cvtClassCov(X, L)
% ŠeƒNƒ‰ƒX‚Ì‹¤•ªU‹‚ß‚é
% input
% X : column vectors
% L : label
% output
% C : Class Cov , DIM x DIM x nClass matrix

nClass = max(L);
C = zeros(size(X,1), size(X,1), nClass);
for I=1:nClass
    C(:,:,I) = cov(X(:,L==I)', 1) * size(X,2);
end

end
