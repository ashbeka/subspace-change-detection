function [L M Score] = cvtK_means(X,k,T,eps)
% function L = k_means(X,k,T)
% X:データ
% k:クラスタ数
% T:繰り返し数
% eps:収束判定

[nDim nNum] = size(X);

if nargin < 4
    eps = 0;
end

%% 初期は位置を決定

% ランダムにラベルをふって平均を計算
L = mod(randperm(nNum),k)+1;
M = zeros(nDim,k);
for I=1:k
   M(:,I) = mean(X(:,L==I),2);
end
D = cvtL2Norm(M,X);
[val,L] = min(D);

% % ランダムに選択し，平均とする
% tmpL = randperm(nNum);
% M = X(:,tmpL(1:k));
% D = cvtL2Norm(M,X);
% [val,L] = min(D);


% 互いに遠いものを初期クラスタとする
% tmpL = randperm(nNum);
% M = zeros(nDim,k);
% M(:,1) = X(:,tmpL(1));
% for I=2:k
%    [val ind] = max(prod(cvtL2Norm(M(:,1:I-1),X),1));
%    M(:,I) = X(:,ind);
% end
% D = cvtL2Norm(M,X);
% [val,L] = min(D);


%%
for t = 1:T
   for I=1:k
      M(:,I) = mean(X(:,L==I),2);
   end
   D = cvtL2Norm(M,X);
   [val,ind] = min(D);
   if ind == L
      break;
   end
   L=ind;
   Score = mean(val);
   if Score <= eps
       break;
   end
end
L=cvtLabelSort(L);
for I=1:k
   M(:,I) = mean(X(:,L==I),2);
end


