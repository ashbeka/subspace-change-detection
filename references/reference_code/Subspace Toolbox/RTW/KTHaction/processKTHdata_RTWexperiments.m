function [processedDATA,Labels] = processKTHdata_RTWexperiments(DATA)
DATA = reshape(DATA,[size(DATA,1)*size(DATA,2)*size(DATA,3),size(DATA,4)]);

% emptyLogic = cellfun(@isempty,DATA);
% idx = ind2sub(size(emptyLogic),find(emptyLogic));

Labels = orzLabel(size(DATA,1),size(DATA,2));

Labels = Labels(~cellfun('isempty',DATA));
DATA = DATA(~cellfun('isempty',DATA));
% for i = 1:13
%     % kth_data(a(1),b(1),c(1),d(1)) = kth_data(a(1),b(1),c(1)+1,d(1));
%     DATA(a(i),b(i),c(i),d(i)) = DATA(a(i),b(i),c(i)+1,d(i));
% end

newsize = [16 16];

DATA2 = cellfun(@(x) rgbvid2grayvid(x),DATA, 'UniformOutput', false);
DATA2 = cellfun(@(x) vidResize(x,newsize),DATA2, 'UniformOutput', false);
processedDATA = cellfun(@(x) orzReshape(x,1),DATA2, 'UniformOutput', false);
end