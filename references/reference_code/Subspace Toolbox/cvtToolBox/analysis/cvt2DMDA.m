function [Y, V, D, J] = cvt2DMDA(X, L, alpha)

L = L(:);
[dim fnum num] = size(X);
label = unique(L);
%クラス数
n_class = size(label, 1);
%クラス間共分散行列
sb = zeros(dim, dim);
%クラス内共分散行列
sw = zeros(dim, dim);
%全平均ベクトル
m = mean(X, 3);

for I=1:n_class;
    z = X(:, :, (L==label(I)));
    Ctmp = cov(reshape(z, dim*fnum, size(z,3))', 1);
    C = zeros(dim, dim);
    for i = fnum
        index = (1:dim) + ((i-1)*dim);
        C = C + Ctmp(index, index);
    end
    C = C / fnum;
    
    sw = sw + C * size(z,3);
    sb = sb + ((m-mean(z,3))*(m-mean(z,3))') * size(z,3);
end
sw = sw / num;
sb = sb / num;

if nargin == 3
    sw = sw + alpha*eye(dim);
end

A = sw\sb;
[V,D] = eigs(A, min(rank(A), n_class-1));

Y = zeros(size(V,2), fnum, num);
for i = 1:num
    Y(:,:,i) = V' * X(:,:,i);
end

D = diag(D);
J =  trace(sb)/trace(sw);
