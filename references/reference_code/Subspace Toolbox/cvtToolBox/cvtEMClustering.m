% function [L M S Score] = cvtEMClustering(X, k, T, eps)
% % function L = k_means(X,k,T)
% % X:データ
% % k:クラスタ数
% % T:繰り返し数
% % eps:収束判定
% 
% [nDim nNum] = size(X);
% 
% if nargin < 4
%     eps = 0;
% end
% 
% %% 初期は位置を決定
% 
% % ランダムにラベルをふって平均を計算
% L = mod(randperm(nNum), k) + 1;
% C = zeros(k, nNum);
% M = zeros(nDim, k);
% S = zeros(nDim, nDim, k);
% 
% C(L + (0:nNum-1)*k) = 1;
% 
% for I = 1:k
%     tmpX = X*C(I,:)';
%     M(:,I) = mean(X, 2);
%     S(:,:,I) = cov(tmpX', 2);
% end
% 
% 
% 
% D = cvtL2Norm(M,X);
% [val, L] = min(D);
% 
% 
% %%
% for t = 1:T
%     for I=1:k
%         M(:,I) = mean(X(:,L==I),2);
%     end
%     D = cvtL2Norm(M,X);
%     [val,ind] = min(D);
%     if ind == L
%         break;
%     end
%     L=ind;
%     Score = mean(val);
%     if Score <= eps
%         break;
%     end
% end
% L=cvtLabelSort(L);
% for I=1:k
%     M(:,I) = mean(X(:,L==I),2);
% end
% 
% 
