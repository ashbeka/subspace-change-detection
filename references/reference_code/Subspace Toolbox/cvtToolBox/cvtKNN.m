function Y = cvtKNN(X, M, L, k)
% K-NNでクラス分類 ver.1.00 by igarashi
% input
%  X: matrix of (dim×NX), 入力列ベクトル集合
%  M: matrix of (dim×NM), 学習列ベクトル集合
%  L: vector of NM-dimension, 学習データのラベル
%  k: scalar, KNNのk
% output
%  Y: vector of NX-dimension, 入力ベクトルの識別結果

Xsize = size(X);
Msize = size(M);
[uL, bL, mL] = unique(L);
n_class = numel(uL);
Y = zeros(1,Xsize(2));
for xi = 1:Xsize(2);
    % 入力ベクトルと学習ベクトル間の距離
    d = cvtL2Norm(X(:,xi), M);    
    
    % 各入力データの近傍k点を投票するhit
    hit = zeros(1, n_class);
    k = min(k, Msize(2));
    % 距離の近い順にデータをソート
    % ソート結果を基に投票
    for i=1:k;
        [tmp I] = min(d, [], 2);
        hit(mL(I)) = hit(mL(I))+1;
        d(I) = Inf;
    end;
    
    [tmp IDX] = max(hit, [], 2);
    Y(xi) = uL(IDX);
end

% % 入力ベクトルと学習ベクトル間の距離
% d = cvtL2Norm(X, M);
% 
% Xsize = size(X);
% Msize = size(M);
% [uL, bL, mL] = unique(L);
% n_class = numel(uL);
% 
% % 各入力データの近傍k点を投票するhit
% hit = zeros(Xsize(2), n_class);
% k = min(k, Msize(2));
% for i=1:k;
%     % 距離の近い順にデータをソート
%     [tmp I] = min(d, [], 2);
%     % ソート結果を基に投票
%     for j=1:Xsize(2);
%         hit(j,mL(I(j))) = hit(j,mL(I(j)))+1;
%         d(j,I(j)) = Inf;
%     end;
% end;
% 
% [tmp IDX] = max(hit, [], 2);
% Y = uL(IDX);
