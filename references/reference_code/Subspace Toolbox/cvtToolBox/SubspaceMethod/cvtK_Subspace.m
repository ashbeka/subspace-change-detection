function [L M Score]= cvtK_Subspace(X,k,nSubDim,T)
% function L = k_means(X,k,T)
% X:データ
% k:クラスタ数
% T:繰り返し数

[nDim nNum] = size(X);
X = cvtNormalize(X);
nSubDim = 2;
%% 初期は位置を決定

% % ランダムに選択し，平均とする
tmpL = randperm(nNum);
M = X(:,tmpL(1:k*nSubDim));
M = reshape(M,nDim,nSubDim,k);
for I=1:k
   M(:,:,I) = orth(M(:,:,I));
end

D = zeros(k,nNum);
for I=1:k
   D(I,:) = cvtVectorNorm( M(:,:,I)'*X);
end

[val,L] = max(D);


%%
for t = 1:T
   %%
   for I=1:k
      tmpX = X(:,L==I);
      [M(:,:,I) tmpD] = eigs(tmpX*tmpX',nSubDim);
   end
   for I=1:k
      D(I,:) = cvtVectorNorm( M(:,:,I)'*X);
   end
   [val,ind] = max(D);
   
   if ind == L
      break;
   end
   L=ind;
   Score = 1-mean(val)
end
L=cvtLabelSort(L);

for I=1:k
   tmpX = X(:,L==I);
   [M(:,:,I) tmpD] = eigs(tmpX*tmpX',nSubDim);
end


