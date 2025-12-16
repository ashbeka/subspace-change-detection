function [F0] = cvtLBPmini(img, s, option)
%   この関数の概要をここに記述
%   詳細説明をここに記述

if ~exist('option', 'var')
    option = 1;
end

Z = double(img);
[h w] = size(Z);

if option == 0
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    X(2,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(3,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    X(4,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(5,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    X(6,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    X(7,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(8,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
%     X=double(X>=0);
    X = double(X>0);
    
    A=reshape([1,2,4,8,16,32,64,128]*X(:,:), h-2*s, w-2*s)+1;
    B = zeros([256*256,size(A)]);
    B((0:numel(A)-1)'*16+A(:)) = 1;
    
elseif option == 1
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
%     X=double(X>=0);
    X=double(X>0);
    
    A=reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s)+1;
    B = zeros([16,size(A)]);
    B((0:numel(A)-1)'*16+A(:)) = 1;
    
elseif option == 2
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    X(2,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    X(3,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
%     X=double(X>=0);
    X=double(X>0);
    
    A=reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s)+1;
    B = zeros([16,size(A)]);
    B((0:numel(A)-1)'*16+A(:)) = 1;
    
end

F0 = sum(B(:,:),2);




