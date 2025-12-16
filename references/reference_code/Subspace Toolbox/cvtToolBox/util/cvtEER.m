function [ P ] = cvtEER(SW, L, step, flg)

L = L(:);
minS1 = min(SW(:));
maxS1 = max(SW(:));

[FAR1 FRR1 A1] = cvtEER_fnc(SW, L, minS1, maxS1, step, flg);

[val1 ind1] = min(abs(FAR1-FRR1));

minS2 = A1(ind1-1);
maxS2 = A1(ind1+1);

[FAR2 FRR2 A2] = cvtEER_fnc(SW, L, minS2, maxS2, step, flg);

[val2 ind2] = min(abs(FAR2-FRR2));

EER = mean([FAR2(ind2) ,FRR2(ind2) ]);

if flg == 1
   [tmp1 tmp2] = max(SW);
else
   [tmp1 tmp2] = min(SW);
end
ER = 1 - mean(tmp2(:) == L(:));

P.EER = EER;
P.THR = A2(ind2);
P.ER = ER;
P.FAR1 = FAR1;
P.FAR2 = FAR2;
P.FRR1 = FRR1;
P.FRR2 = FRR2;
P.A1 = A1;
P.A2 = A2;


function [ mFAR mFRR A ] = cvtEER_fnc(SW, L, minS, maxS, step, flg)

n_class = numel(unique(L));
w_step = (maxS-minS)/(step-2) + eps;
A = (minS - w_step):w_step:(maxS+w_step);
FRR = zeros(n_class,step);
FAR = zeros(n_class,step);
for a = 1:size(A,2);
    
    if flg == 1
        F = (SW <= A(a));
        G = (SW >= A(a));
    else
        F = (SW >= A(a));
        G = (SW <= A(a));
    end        
        
    %本人類似度が閾値以下の試行回数／全試行回数
    for I=1:n_class;
        FRR(I,a) = mean(F(I,L==I));
    end

    %他人類似度が閾値以下の試行回数／(全試行回数-本人の試行回数)
    for I=1:n_class;
        tmp = G((1:n_class)~=I,L==I);
        FAR(I,a) = mean(tmp(:));
    end
    
end
mFAR = mean(FAR,1);
mFRR = mean(FRR,1);
