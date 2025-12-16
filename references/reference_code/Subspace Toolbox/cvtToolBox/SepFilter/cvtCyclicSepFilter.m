function mu = cvtCyclicSepFilter(V1, V2)
% bases on von mises distribution
%

N1 = size(V1,2);
N2 = size(V2,2);
N = N1 + N2;

V1sum = sum(V1,2);
V2sum = sum(V2,2);
Vave = (V1sum + V2sum) / N;

St = 1 - norm(Vave,2);

if St < 1e-10
    mu = 0;
else
    S1 = 1 - norm(V1sum/N1, 2);
    S2 = 1 - norm(V2sum/N2, 2);
    Sw = (N1 * S1 + N2 * S2) / N;
    Sb = St - Sw; 
    mu = Sb / St;
end
end
