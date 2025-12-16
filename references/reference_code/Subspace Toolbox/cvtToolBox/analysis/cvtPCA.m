function [Y, V, D, K] = cvtPCA(X, sdim_or_contribution_rate, alpha)
% 主成分分析 ver.1.00 by ohkawa
% input
%  X: matrix of (m×n), column vectors
%  [sdim]: scalar, 必要な主軸の数, 省略すると全本数
% output
%  Y: matrix of (sdim×n), 各データの主軸に対するパラメータ表示
%  V: matrix of (m×sdim), 主軸
%  D: vector of sdim-dimension, 固有値
%  K: scalar, 累積寄与率

% 正則化パラメータなし
if(nargin < 3)
    alpha = 0; 
end

C = cov(X',1);
C = C + alpha * eye(size(C));

% 寄与率で次元を決定
if(sdim_or_contribution_rate < 1)
    [V D] = cvtEig(C);
    sdim = find(cumsum(D) > sum(D)*sdim_or_contribution_rate, 1);
    V = V(:,1:sdim);
    D = D(1:sdim);
        
else
    % 何も指定されていない場合
    if(nargin < 2)
        [V D] = cvtEig(C);
    %指定の次元
    else
        [V D] = cvtEig(C, sdim_or_contribution_rate);
    end
end

% [V,tmpD]= svd(C, 0);
% D = diag(tmpD(1:sdim));
% V = V(:,1:sdim);


Y = V' * X;
K = sum(D)/trace(C);
