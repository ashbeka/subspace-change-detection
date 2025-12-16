function [V D] = cvtEig(A, dim)
% 固有値と固有ベクトルを降順にソートして返す

[ tmpV tmpD ] = eig(A);
tmpD = real(diag(tmpD));
tmpV = real(tmpV);

[val, index]= sort(real(tmpD),'descend');
D = tmpD(index);
V = tmpV(:,index);

if nargin > 2
    D = D(1:dim);
    V = V(:,1:dim);
end