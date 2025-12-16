function D = orzL2Distance(A,B)

if size(size(A),2) >2
    A=A(:,:);
end
if size(size(B),2) >2
    B=B(:,:);
end
nNumA = size(A,2);
nNumB = size(B,2);
D = abs(repmat(sum((A.^2),1)',1,nNumB)+repmat(sum((B.^2),1),nNumA,1)-2*A'*B);

%function d = orzL2Distance(X,Y)
% Euclidean distance matrix between column vectors in X and Y.
%                                   from knn_old.m in somtoolbox
%				    ver. 1.00
% d(i, j) = norm(x(i) - y(j))^2
%X = X(:,:)';
%Y = Y(:,:)';
%U=~isnan(Y); 
%Y(~U)=0;
%V=~isnan(X); 
%X(~V)=0;
%d = abs(X.^2*U'+V*Y'.^2-2*X*Y');