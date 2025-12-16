
function Y = cvtShiftOrigin(X, Origin)
% 列ベクトル集合をOriginを原点として移動 ver.1.00 by nosaka
% input
%  X: column vectors
% output
%  Y: 移動したの列ベクトル集合
Y = double(X) - repmat(Origin, 1, size(X,2));
