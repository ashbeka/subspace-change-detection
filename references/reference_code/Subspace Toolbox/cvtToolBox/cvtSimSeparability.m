function SEP = cvtSimSeparability(SIM,L)

[nClass nNum] = size(SIM);


W = zeros(1,nNum);
B = zeros(nClass-1,nNum);

for I=1:nNum
    tmpSIM=SIM(:,I);
    W(1,I) = tmpSIM(L(I)==(1:nClass));
    B(:,I) = tmpSIM(L(I)~=(1:nClass));    
end

B=B(:)';

Sb = size(B,2)*(mean(B) -mean([B,W]))^2 + size(W,2)*(mean(W) -mean([B,W]))^2;
St = (size(B,2)+size(W,2)) * var([B,W],1);
SEP = Sb/St;