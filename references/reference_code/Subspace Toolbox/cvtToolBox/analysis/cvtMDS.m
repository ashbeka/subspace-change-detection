function [Y] = cvtMDS(D, k)
% 多次元尺度構成法 [MultiDimensional Scaling]
% D : NxNの距離行列
% k : 小さくしたときの次元
% 
% 2010/09/08
% written by Nosaka Ryusuke

N = size(D,1);
J = eye(N) - ones(N)/N;
P = - 0.5 * J * D.^2 * J';
[U V] = eigs(P, k);
Y = (U * sqrt(abs(V)))';

end
