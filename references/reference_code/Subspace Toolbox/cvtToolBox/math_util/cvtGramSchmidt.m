
function eigen_v = cvtGramSchmidt(data, n)
% グラムシュミットの正規直交化法 ver.1.00 by igarashi
% nが無い場合はフルランクで計算するが、その場合orth(data)の方が高速
% input
%  data: column vectors
%  [n]: 出力する基底の数
% output
%  eigen_v: n本の正規直交基底[dim, num] = size(data);

dim = size(data,1);
if(nargin < 2)
    n = rank(data);
    eigen_v = zeros(dim, n);
else
    eigen_v = zeros(dim, n);
end

eigen_v(:,1) = data(:,1) ./ norm(data(:,1));
for I=2:n;
    v = data(:,I);
    for J=1:I-1;
        v = v - (eigen_v(:,J)'*data(:,I))*eigen_v(:,J);
    end;
    eigen_v(:,I) = v ./ norm(v);
end;