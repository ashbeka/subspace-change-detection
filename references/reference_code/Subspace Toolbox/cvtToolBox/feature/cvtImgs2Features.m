function cvtImgs2Features(imgs, funcs_feature, dims_feature, dir_path, color_option)
% images to features mat-file
% color_option : 'c' -> color image
%


if ~exist('dir_path', 'var')
    dir_path = '.';
end


% color image
if exist('color_option', 'var') && color_option == 'c'

    s = size(imgs);
    imgs = imgs(:,:,:,:);
    nImg = size(imgs,4);

    for fi = 1:numel(funcs_feature)

        func_feature = funcs_feature{fi};
        disp(func_feature);

        filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
        if exist(filename, 'file'), continue,  end
        data_x = zeros(dims_feature(fi), nImg);

        for iImg = 1:nImg
            data_x(:, iImg) = func_feature(imgs(:,:,:,iImg));
        end

        data_x = reshape(data_x, [size(data_x,1), s(4:end)]);
        save(filename, 'data_x', 'func_feature');
    end
    
% gray image
else
    s = size(imgs);
    imgs = imgs(:,:,:);
    nImg = size(imgs,3);

    for fi = 1:numel(funcs_feature)

        func_feature = funcs_feature{fi};
        disp(func_feature);

        filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
        if exist(filename, 'file'), continue,  end
        data_x = zeros(dims_feature(fi), nImg);

        for iImg = 1:nImg
            data_x(:, iImg) = func_feature(imgs(:,:,iImg));
        end

        data_x = reshape(data_x, [size(data_x,1), s(3:end)]);
        save(filename, 'data_x', 'func_feature');
    end

    
end

% if ndims(imgs) == 3
%     [h w num] = size(imgs);
%     
%     for fi = 1:numel(funcs_feature)
%         data_x = zeros(dims_feature(fi), num);
%         func_feature = funcs_feature{fi};
%         disp(func_feature);
%         filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
%         
%         if exist(filename, 'file'), continue,  end
%         
%         for ii = 1:num
%             data_x(:, ii) = func_feature(imgs(:,:,ii));
%         end
%         save(filename, 'data_x', 'func_feature');
%     end
%     
%     
% elseif ndims(imgs) == 4
%     [h w num num_class] = size(imgs);
%     
%     for fi = 1:numel(funcs_feature)
%         func_feature = funcs_feature{fi};
%         disp(func_feature);
%         
%         filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
%         if exist(filename, 'file'), continue,  end
%         
%         data_x = zeros(dims_feature(fi), num, num_class);
%         for ci = 1:num_class
%         for ii = 1:num
%                 data_x(:, ii, ci) = func_feature(imgs(:,:,ii,ci));
%         end
%         end
% 
%         save(filename, 'data_x', 'func_feature');
%     end
%     
% elseif ndims(imgs) == 5
%     [h w color num num_class] = size(imgs);
%     
%     for fi = 1:numel(funcs_feature)
%         data_x = zeros(dims_feature(fi), num, num_class);
%         func_feature = funcs_feature{fi};
%         disp(func_feature);
% 
%         filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
%         if exist(filename, 'file'), continue,  end
%         
%         for ci = 1:num_class
%             for ii = 1:num
%                 data_x(:, ii, ci) = func_feature(imgs(:,:,ii,ci));
%             end
%         end
% 
%         save(filename, 'data_x', 'func_feature');
%     end
%     
% %% ó·äOÅIÅI
% elseif ndims(imgs) == 6
%     [h w nk1 nk2 nk3 nk4] = size(imgs);
%     matlabpool;
%     for fi = 1:numel(funcs_feature)
%         data_x = zeros(dims_feature(fi), nk1,nk2,nk3,nk4);
%         func_feature = funcs_feature{fi};
%         disp(func_feature);
%         filename = fullfile(dir_path,  [func2str(func_feature) '.mat']);
%         if exist(filename, 'file'), continue,  end
%         
%         for ik4 = 1:nk4
%         for ik3 = 1:nk3
%         for ik2 = 1:nk2
%         for ik1 = 1:nk1
%             data_x(:, ik1,ik2,ik3,ik4) = func_feature(imgs(:,:,ik1,ik2,ik3,ik4));
%         end
%         end
%         end
%         end
%         save(filename, 'data_x', 'func_feature');
%     end
%     matlabpool close;
% end


end
