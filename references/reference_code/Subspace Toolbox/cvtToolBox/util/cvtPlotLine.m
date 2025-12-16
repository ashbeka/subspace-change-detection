function cvtPlotLine(X,Y,No)

colors{1} = 'r';
colors{2} = 'g';
colors{3} = 'k';
colors{4} = 'm';
colors{5} = 'c';
colors{6} = 'b';

lines{1} = '-';
lines{2} = '--';
lines{3} = ':';
lines{4} = '-.';
markers = cell(size(lines,2)*size(colors,2),1);
for J2 = 1:size(lines,2)   
    for J1 = 1:size(colors,2)
        markers{size(colors,2)*(J2-1) + J1} = strcat(colors{J1},lines{J2});
    end
end

% draw
if(nargin<3)
    figure(100);
else
    figure(No);
    clf(No);
end
hold on;
for I=1:size(Y,2)
    plot(X,Y(:,I),markers{I});
end
% ylim([0.5 1]);
ylim([0 1]);
hold off;
