function strs = cvtSprintfs(str, varargin)
%
%  cvtSprintfs('%d*x*%d_%s', 1:3, 3:5, {'124', '123123'})
% 


num_var = numel(varargin);
vars = cell(num_var, 1);
num_param = 1;
for vi = 1:num_var
   num_param = num_param * numel(varargin{vi}); 
    if isa(varargin{vi}, 'cell')
        vars{vi} = varargin{vi}; 
    else
       vars{vi} = num2cell(varargin{vi}); 
    end
end


indexes = ones(num_var,1);

indexes(end) = 0;

params = cell(num_var,1);
strs = cell(num_param, 1);
for pi = 1:num_param
    % indexes 更新
    indexes(end) = indexes(end) + 1;
    for vi = num_var:-1:1
      if indexes(vi) > numel(vars{vi});
          indexes(vi) = 1;
          indexes(vi-1) = indexes(vi-1) + 1;
      end
    end

    % params 更新
    for vi = 1:num_var
        params{vi} = vars{vi}{indexes(vi)};
    end
   
    % 文字列取得
    strs{pi} = sprintf(str, params{:});
end
