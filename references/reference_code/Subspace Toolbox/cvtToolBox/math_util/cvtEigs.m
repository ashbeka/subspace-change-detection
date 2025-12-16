function [V D] = cvtEigs(A, dim)
% 固有値と固有ベクトルを降順にソートして返す
if nargin < 2
    [ tmpV tmpD ] = eigs(A);
else
    [ tmpV tmpD ] = eigs(A,dim);
end

tmpD = diag(tmpD);
[val, index]= sort(real(tmpD),'descend');
D = tmpD(index);
V = tmpV(:,index);
