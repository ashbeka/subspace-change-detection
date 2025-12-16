function mu = cvtSepFilter(V1, V2)
%
%

V1 = V1(:);
V2 = V2(:);

N1 = numel(V1);
N2 = numel(V2);
N = N1 + N2;


M = mean([V1;V2]);
St = cov([V1;V2],1);

if St <= 1e-5
    mu = 0;
else
    M1 = mean(V1);
    M2 = mean(V2);
    Sb = (N1 * (M1 - M)^2 + N2 * (M2 - M)^2) / N;
    mu = Sb / St;
end

end
