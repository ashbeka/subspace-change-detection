function [G Z] = cvtWhiteningBasis(X,pAlpha)

X = X(:,:,:);
[nDim nSubDim nClass] = size(X);
Z = cvtNormalize(X);
G = eye(nDim,nDim);
nCnt=0;
while 1
    [B C] = cvtEig(cvtAutoCorrMat(Z));
    O=B*diag(sqrt(1./C))*B';
    for I=1:nClass
        Z(:,:,I) = orth(O*Z(:,:,I));
    end    
    G =O*G;
    ZZ =  Z(:,:)*Z(:,:)';
    ZZ = (ZZ/norm(ZZ,'fro'))*sqrt(nDim);
    norm(ZZ-eye(nDim),'fro')
    if norm(ZZ-eye(nDim),'fro') < 1e-5 
        break
    end
    if nCnt > 100
        break
    end
    nCnt  = nCnt+1;
end
G = G/norm(G,'fro');
    
