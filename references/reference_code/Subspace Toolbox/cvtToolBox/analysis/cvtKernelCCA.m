function [Y1, Y2, A1, A2, D] = cvtKernelCCA(X1, X2, k, sigma2, t1, t2)
% カーネル正準相関分析　[Kernel Canonical Correlation Analys]
% Input
% X1:InputDim x DataNum matrix
% X2:InputDim x DataNum matrix
% sigma2: Kernel Parameters
% t1, t2: Regularisation Parameters
% Output
% Y1:mapped X1
% Y2:mapped X2
% A1:map matrix X1
% A2:map matrix X2
% D :canonical angle cos^2 ???
% k :eigs num
% Written by nosaka
% Reference to "Kernel Method for Pattern Analysis"

num1 = size(X1, 2);
num2 = size(X2, 2);
K1 = cvtKernelMatrix(X1, sigma2);
K2 = cvtKernelMatrix(X2, sigma2);

K12 = K1*K2;

A = [zeros(num1, num2), K12; zeros(num2, num1), K12'];
B = [(1-t1)*K1*K1+t1*K1, zeros(num1,num2); zeros(num2,num1), (1-t2)*K2*K2+t2*K2];

[V, D] = eigs(A, B, k, 'LM');

A1 = V(1:num1, :);
A2 = V((num1+1):end, :);

Y1 = A1'*K1;
Y2 = A2'*K2;

% R1 = chol(K1);
% R2 = chol(K2);
% 
% Tmp1 = (1-t1) * (R1*R1') + t1 * eye(size(R1,1));
% Tmp2 = (1-t2) * (R2*R2') + t2 * eye(size(R2,1));
% 
% R = chol(Tmp2);
% 
% [V, L] = eig(R' \ R2 * R1' / Tmp1 * R1 * R2' / R);
% 
% A2 = R \ V;
% A2 = cvtNormalize(A2);
% A1 = L \ Tmp1 \ R1 * R2' * A2;
% A1 = cvtNormalize(A1);
% 
% D = L;

end
