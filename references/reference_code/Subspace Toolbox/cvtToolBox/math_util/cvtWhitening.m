function [O Y] = cvtWhitening(X,pAlpha)

[B C] = eig( cvtAutoCorrMat(X) + pAlpha * eye(size(X,1)));
O = sqrt(inv(C))*B';
Y = reshape(O*X(:,:),size(X));
