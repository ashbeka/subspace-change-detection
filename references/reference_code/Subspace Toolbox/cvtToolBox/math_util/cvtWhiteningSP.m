function [G Y] = cvtWhiteningSP(X,pAlpha)

nDim = size(X,1);
Z = cvtNormalize(X);
G = eye(nDim,nDim);
while 1    
    [O Z] = cvtWhitening(Z,0);
    Z = cvtNormalize(Z);
    G =O*G;
    G = G/norm(G,'fro');    
    ZZ =  Z*Z' *(nDim/size(Z,2));
    lsl = norm(ZZ-eye(nDim),'fro') 
    if lsl < 1e-5
        break
    end    
end
Y = cvtNormalize( G * X);

