function [G] = cvtWhiteningSP2(X)

nDim = size(X,1);
Z = cvtNormalize(X);
G = eye(nDim,nDim);
while 1
   [B C] = eig(cvtAutoCorrMat(Z));
   O = B*sqrt(inv(C))*B';
   Z = cvtNormalize(O*Z);
   G =O*G;
   G = G/norm(G,'fro');
   ZZ =  Z*Z' *(nDim/size(Z,2));
   lsl = norm(ZZ-eye(nDim),'fro')
   if lsl < 1e-5
      break
   end
end


