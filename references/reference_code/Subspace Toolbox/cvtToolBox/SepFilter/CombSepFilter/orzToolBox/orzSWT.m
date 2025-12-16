function [S,G,CR] = orzSWT(X,cRate)
% [S,G,CR] = orzSWT(X,cRate)
% 球面白色化変換（Shperical Whitening Transform）
% input
%  X: 入力マトリックス
%  cRate: 寄与率
%
% output
%  Z: 変換されたデータ
%  G: 変換行列
%  C: 寄与率


X = X(:,:);
[Y,U,D,CR] = orzPCA(orzNormalize(X),cRate,'R');
[nDim,nNum] = size(Y);
Z = orzNormalize(Y);
G = eye(nDim,nDim);
while 1
    R = Z*Z'/nNum;
    A = norm(R-(1/(nDim))*eye(nDim),'fro');
  %  disp(A)
    if A < 1e-13
        break
    end
    [B C] = eig(R);
    O = sqrt(inv(C))*B';
    Z = orzNormalize(O*Z);
    G =O*G;
end
G = G*U';
S = orzNormalize(G*X);

