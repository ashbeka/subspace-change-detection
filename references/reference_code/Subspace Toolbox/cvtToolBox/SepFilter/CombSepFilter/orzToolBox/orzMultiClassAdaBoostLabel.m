function [Alp No] = orzMultiClassAdaBoostLabel(ANS,Label,nTry)
% Multi-class AdaBoost Ji Zhu

nClass = length(unique(Label));
[nC nProb ] = size(ANS);
W = ones(1,nProb)/nProb;
Err = zeros(nC,nTry);
No  = zeros(1,nTry);
Alp = zeros(1,nTry);
IND = false(nC,nProb);

for M=1:nC
    IND(M,:) = Label ~=ANS(M,:);
end

for T = 1:nTry
    Err(:,T) = sum(IND.*repmat(W,nC,1),2)/sum(W);    
    [a b] = min(Err(:,T));
    No(T) = b;    
    Alp(T) = log((1-a)/a) + log(nClass-1);
    if Alp(T) == Inf
        Alp(T) =1;
        Alp = Alp(1:T);
        No = No(1:T);
        break;
    end    
    W=W.*exp(Alp(T) * (IND(b,:)));    
    W = W/sum(W);  
end
