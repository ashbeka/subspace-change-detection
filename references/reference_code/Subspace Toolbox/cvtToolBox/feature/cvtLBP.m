function [histogram indexes] = cvtLBP(img, s, config)
% cvtLBP
% histgram : 256[dim]
% ステップ幅が画像をはみ出す場合はエラーが出力されないので，注意．

if ~isa('img', 'double')
    Z = im2double(img);
end
[h w] = size(Z);


if ~exist('s', 'var')
    s = 1;
end
if ~exist('config', 'var')
    config=0;
end

if config == 0      % ８方向 uniform
    X = zeros(8,h-2*s,w-2*s);
    C        = Z(1+s  :h-s   ,1+s :w-s  );
    X(1,:,:) = Z(1    :h-s-s,1    :w-s-s) - C;
    X(2,:,:) = Z(1+s  :h-s  ,1    :w-s-s) - C;
    X(3,:,:) = Z(1+s+s:h    ,1    :w-s-s) - C;
    X(4,:,:) = Z(1    :h-s-s,1+s  :w-s  ) - C;
    X(5,:,:) = Z(1+s+s:h    ,1+s  :w-s  ) - C;
    X(6,:,:) = Z(1    :h-s-s,1+s+s:w    ) - C;
    X(7,:,:) = Z(1+s  :h-s  ,1+s+s:w    ) - C;
    X(8,:,:) = Z(1+s+s:h    ,1+s+s:w    ) - C;
    X=double(X>0);
    
    indexes = [1,2,4,8,16,32,64,128]*X(:,:);
    histogram = hist(indexes, 0:255);
    histogram = histogram(:);
    
elseif config == 1    % 上下左右
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    X=double(X>0);
    
    indexes = [1,2,4,8]*X(:,:);
    histogram = hist(indexes, 0:15);
    histogram = histogram(:);

else    %斜め４つ
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    X(2,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    X(3,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
    X=double(X>0);
    
    indexes = [1,2,4,8]*X(:,:);
    histogram = hist(indexes, 0:15);
    histogram = histogram(:);
end

end
