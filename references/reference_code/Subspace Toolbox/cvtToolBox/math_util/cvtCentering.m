
function [Y, M] = cvtCentering(X)
% 列ベクトル集合の平均を原点へ移動 ver.1.00 by ohkawa
% input
%  X: column vectors
% output
%  Y: 平均が0の列ベクトル集合
%  M: Xの平均
M = mean(X,2);
Y = double(X) - repmat(M, 1, size(X,2));
