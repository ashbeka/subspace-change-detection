function cvtPlot3(X, L, No)
% Xのデータを主成分分析で得られる上位3軸にプロットする ver.1.00 by ohkawa
% input
%  X: matrix of (m×n), column vectors
%  [L]: vector of n-dimension, labels
%  [No]: figure number

if(nargin<2)
    L = ones(1,size(X,2));
end

X = X(:,:);
L = L(:)';
dim = size(X,1);
uL = unique(L);
n_class = numel(uL);
% if n_class > (6*13)
%     error('クラス数多すぎ')
% end

% colors = ['r.';'g+';'ko';'cs';'mv';'bp'];
% colors = ['r.';'g.';'b.';'k.';'c.';'m.'; ...
%     'ro';'go';'bo';'ko';'co';'mo'; ...
%     'r+';'g+';'b+';'k+';'c+';'m+'; ...
%     'rx';'gx';'bx';'kx';'cx';'mx'; ...
%     'r*';'g*';'b*';'k*';'c*';'m*'; ...
%     'rs';'gs';'bs';'ks';'cs';'ms'; ...
%     'rd';'gd';'bd';'kd';'cd';'md'; ...
%     'rv';'gv';'bv';'kv';'cv';'mv'; ...
%     'r^';'g^';'b^';'k^';'c^';'m^'; ...
%     'r<';'g<';'b<';'k<';'c<';'m<'; ...
%     'r>';'g>';'b>';'k>';'c>';'m>'; ...
%     'rp';'gp';'bp';'kp';'cp';'mp'; ...
%     'rh';'gh';'bh';'kh';'ch';'mh';];
% colors = hsv(n_class/2);
% colors = hsv(9);
colors = hsv(n_class);
marker = ['+', 'o','d'];

if dim > 3;
    Y = cvtKLE(X,3);
    Y = real(Y);
elseif dim ==2;
    Y = X;
    Y(3,:) = 0;
elseif dim ==1;
    Y = X;
    Y(2,:) = 0;
    Y(3,:) = 0;
else
    Y = X;
end

% draw
if(nargin<3)
    figure;
else
    figure(No);
    clf(No);
end

hold on;
for i=1:n_class
%     if(i <= n_class/2)
%     if(i <= 9)
        plot3(Y(1,L==uL(i)), Y(2,L==uL(i)), Y(3,L==uL(i)), marker(1), 'Color', colors(i,:));
%     else
%         plot3(Y(1,L==uL(i)), Y(2,L==uL(i)), Y(3,L==uL(i)), marker(2), 'Color', colors(i-9,:));
%         plot3(Y(1,L==uL(i)), Y(2,L==uL(i)), Y(3,L==uL(i)), marker(2), 'Color', colors(i-n_class/2,:));
%     end
end

hold off;
