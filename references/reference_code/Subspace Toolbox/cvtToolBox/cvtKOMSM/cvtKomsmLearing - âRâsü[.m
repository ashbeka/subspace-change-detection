function [ KO, A,C_RATE,beta ] = cvtKomsmLearing( X , L, P)

%% Constract Kernel Matrix
K = cvtKernelMatrix(X,P.sigma2);

%% Kernel PCA is done to each class.
C_RATE = zeros(1,P.n_class);
for l = 1:P.n_class;
    waitbar(l/P.n_class);
    t_K = K(L==l,L==l);
    size(t_K)
    [t_A,tmp] = eigs(t_K,P.dim_1);
    t_eval = diag(tmp);
    for e=1:P.dim_1,
        t_A(:,e) = t_A(:,e)/(sqrt(t_eval(e)));
    end;
    A(l).A = t_A;
    C_RATE(l) = sum(t_eval)/trace(t_K);
end


%% Constract Kernel Orthogonal Matrix : KO
KO = zeros(P.dim_2,P.dim_2);
D = zeros(P.dim_2,P.dim_2);
for c1 = 1:P.n_class
    for c2 = 1:P.n_class
        tmpK = K(L==c1,L==c2);
        A1 = A(c1).A;
        A2 = A(c2).A;
        s1 = (c1-1)*P.dim_1+1:c1*P.dim_1;
        s2 = (c2-1)*P.dim_1+1:c2*P.dim_1;
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
