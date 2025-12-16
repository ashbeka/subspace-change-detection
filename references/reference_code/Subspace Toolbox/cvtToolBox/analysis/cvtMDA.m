function [Y, V, D, J] = cvtMDA(X,L, contributing_rate, alpha)
% 重判別分析 ver.1.00 by ohkawa
% input
%  X: matrix of (dim×m), column vectors
%  L: vector of m-dimension, L(i) is label of X(:,i).
%
% output
%  Y: 判別空間へ射影されたX
%  V: 判別軸
%  J: 分離度

L = L(:);
dim = size(X,1);
label = unique(L);
%クラス数
n_class = size(label,1);
%クラス間共分散行列
sb = zeros(dim,dim);
%クラス内共分散行列
sw = zeros(dim,dim);
%全平均ベクトル
m = mean(X,2);

for I=1:n_class;
    z = X(:,(L==label(I)));
    sw = sw + cov(z',1) * (size(z,2)/size(X,2));
    sb = sb + (((m-mean(z,2))*(m-mean(z,2))') * (size(z,2)/size(X,2)));
end

if nargin == 4
    if(isfloat(alpha) || isinteger(alpha))
        sw = sw + alpha*eye(dim);
    else        
        alpha = 1 / max(svd(sw));
       sw = sw +  alpha * eye(dim);
    end
end

[Uw Dw] = svd(sw, 'econ');
Dw = diag(Dw);
dimw = find(cumsum(Dw) > sum(Dw)*contributing_rate, 1);
Uw = Uw(:,1:dimw).*repmat(1./sqrt(Dw(1:dimw))', size(Uw,1), 1);

[Ub Db] = svd(Uw'*sb, 'econ');
V = Uw*Ub;
D = Db;

%A = inv(sw)*sb;
% A = sw\sb;
% [V,D] = cvtEig(A,min(n_class-1, rank(A)));
% [V,D] = eigs(A,3);
% if r >= (n_class-1)
%     [V, D, ~] = svds(A, n_class);
%     V = V(:,1:(n_class-1));
% else
%     [V, D, ~] = svds(A, r);
% end
Y = V'*X;
D = diag(D);
J =  trace(sb)/trace(sw);
