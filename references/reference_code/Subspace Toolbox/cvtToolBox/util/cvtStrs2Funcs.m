function funcs = cvtStrs2Funcs(strs)
% 
% strs = cvtSprintfs('@(x) %d*x*%d*%s', 1:3, 3:5, {'124', '123123'});
% funcs = cvtStrs2Funcs(strs);
%

strs = strs(:);

funcs = cell(numel(strs), 1);

for si = 1:numel(strs); 
    funcs{si} = str2func(strs{si});
end
