function [L M Score]= cvtK_medoid(X,k,T)
% function L = k_means(X,k,T)
% X:データ
% k:クラスタ数
% T:繰り返し数

[nDim nNum] = size(X);

%% 初期は位置を決定

% % % ランダムに選択し，平均とする
tmpL = randperm(nNum);
M = X(:,tmpL(1:k));
D = cvtL2Norm(M,X);
[val,L] = min(D);


%互いに遠いものを初期クラスタとする
% tmpL = randperm(nNum);
% M = zeros(nDim,k);
% M(:,1) = X(:,tmpL(1));
% for I=2:k
%    [val ind] = max(prod(cvtL2Norm(M(:,1:I-1),X),1));
%    M(:,I) = X(:,ind);
% end
% D = cvtL2Norm(M,X);
% [val,L] = min(D);

A = sqrt(cvtL2Norm(X,X));


%%
for t = 1:T
   for I=1:k
      [val index ] = min(mean(A(L==I,L==I),2));
      tmpX = X(:,L==I);
      M(:,I) = tmpX(:,index);
   end
   
   D = cvtL2Norm(M,X);
   [val,ind] = min(D);
   Score = mean(val);
   if ind == L
      break;
   end
   L=ind;   
end

L=cvtLabelSort(L);
for I=1:k
   M(:,I) = mean(X(:,L==I),2);
end


