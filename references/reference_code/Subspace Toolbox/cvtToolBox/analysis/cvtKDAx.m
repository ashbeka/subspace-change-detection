function [Y,A,M] = cvtKDAx(X, L, sigma2, alpha)
% Kernel Multiple Discriminant Analysis
% X      : Training data
% L      : Training Label
% sigma2 : parameter of gaussian kernel
% alpha  : white noise


uL = unique(L);
N = numel(L);
n_class = numel(uL);
kc = zeros(N,n_class);


Nc = zeros(1,n_class);
for i=1:n_class;
    Nc(i) = sum(L==uL(i));
end
N= sum(Nc);

K = cvtKernelMatrix(X,sigma2);
[Y,A,M] = orzMDA(K,L,alpha);
