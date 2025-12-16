function Y = cvtVectorNorm(X)
% Xの列ベクトルに対するL2ノルムを計算する
Y = sqrt(sum(X.^2,1));