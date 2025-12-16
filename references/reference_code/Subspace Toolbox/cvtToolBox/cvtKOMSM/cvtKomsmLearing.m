function [ KO, A,C_RATE,beta ] = cvtKomsmLearing( X , L, P)
%[ KO, A,C_RATE,beta ] = cvtKomsmLearing( X , L, P)
%% Constract Kernel Matrix
%K = cvtKernelMatrixLowMemory(X,P.Sigma2);
K = cvtKernelMatrix(X,P.Sigma2);

%% Kernel PCA is done to each class.
C_RATE = zeros(1,P.nClass);
for l = 1:P.nClass;
    t_K = K(L==l,L==l);
    [t_A,tmp] = eigs(t_K,P.nOrthSubDim);
    t_eval = diag(tmp);
    for e=1:P.nOrthSubDim,
        t_A(:,e) = t_A(:,e)/(sqrt(t_eval(e)));
    end;
    A(l).A = t_A;
    C_RATE(l) = sum(t_eval)/trace(t_K);
end


%% Constract Kernel Orthogonal Matrix : KO
KO = zeros(P.nOrthDim,P.nOrthDim);
D = zeros(P.nOrthDim,P.nOrthDim);
for c1 = 1:P.nClass
    for c2 = 1:P.nClass
        tmpK = K(L==c1,L==c2);
        A1 = A(c1).A;
        A2 = A(c2).A;
        s1 = (c1-1)*P.nOrthSubDim+1:c1*P.nOrthSubDim;
        s2 = (c2-1)*P.nOrthSubDim+1:c2*P.nOrthSubDim;
        tmpD = A1' * tmpK *A2;
        D(s1,s2) = tmpD;
        D(s2,s1) = tmpD';
    end
end

[b,tmp] = eig(D);
[beta, ind] = sort(diag(tmp),'descend');
b = b(:,ind);
for i=1:size(D,2),
    KO(:,i) = b(:,i)/(beta(i));
end;
