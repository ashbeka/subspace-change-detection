function [L M Score]= cvtK_meansSpherical(X,k,T)
% function L = k_means(X,k,T)
% X:データ
% k:クラスタ数
% T:繰り返し数

[nDim nNum] = size(X);
X = cvtNormalize(X);

%% 初期は位置を決定

% ランダムにラベルをふって平均を計算
% L = mod(randperm(nNum),k)+1;
%M = zeros(nDim,k);
% for I=1:k
%    M(:,I) = mean(X(:,L==I),2);
% end
% D = cvtL2Norm(M,X);
% [val,L] = min(D);

% % ランダムに選択し，平均とする
% tmpL = randperm(nNum);
% M = X(:,tmpL(1:k));
% D = M'*X;
% [val,L] = max(D);


% 互いに遠いものを初期クラスタとする
tmpL = randperm(nNum);
M = zeros(nDim,k);
M(:,1) = X(:,tmpL(1));
for I=2:k
   [val ind] = max(prod(cvtL2Norm(M(:,1:I-1),X),1));
   M(:,I) = X(:,ind);
end
D = cvtL2Norm(M,X);
[val,L] = min(D);


%%
for t = 1:T
   for I=1:k
      M(:,I) = mean(X(:,L==I),2);
   end
   M = cvtNormalize(M);
   D = M'*X;
   [val,ind] = max(D);
   
   if ind == L
      break;
   end
   L=ind;
   Score = 1 - mean(val);
  
end
L=cvtLabelSort(L);
for I=1:k
   M(:,I) = mean(X(:,L==I),2);
end

