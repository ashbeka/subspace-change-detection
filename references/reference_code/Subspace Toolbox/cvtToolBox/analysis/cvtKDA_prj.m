function [Z] = cvtKDA_prj(T,X,A, sigma2)
% Kernel Multiple Discriminant Analysis
% T      : Testing data
% X      : Training data
% A      : Coefficient matrix
% sigma2 : Parameter of gaussian kernel

kkT = cvtKernelMatrix(X,T,sigma2);
% N = size(X,2);
% Nt = size(T,2);
% kkT = zeros(N,Nt);
% for i = 1:Nt;
%     for j = 1:N;
%         x = X(:,j)-T(:,i);
%         kkT(j,i) = exp(-((x'*x)/sigma2));
%     end
% end
Z = A'*kkT;