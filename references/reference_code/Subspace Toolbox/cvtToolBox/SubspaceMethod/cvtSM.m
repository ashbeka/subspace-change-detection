function [ SIM ] = cvtSM(Y,V)
%function [ SIM ] = cvtSM(Y,L,V)
[dim sdim n_class] = size(V);
SIM = zeros(n_class,size(Y,2));
for I = 1:n_class
   SIM(I,:) = sum((V(:,:,I)'*Y).^2,1);
end