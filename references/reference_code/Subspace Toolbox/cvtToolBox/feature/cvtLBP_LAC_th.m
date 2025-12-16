function [ F1 F0 ] = cvtLBP_LAC_th(img, s, r, config, th)
%   ステップ幅が画像をはみ出す場合はエラーが出力される．
%   

if ~exist('config', 'var')
    config = 1;
end

Z = double(img);
[h w] = size(Z);

if config == 0
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
    [hh ww] = size(A);
    D  = (A(1+r  :hh-r  ,1+r  :ww-r) - 1) * 256;
    Y1 = A(1+r  :hh-r  ,1+r+r:ww-r+r) + D;
    Y2 = A(1+r-r:hh-r-r,1+r+r:ww-r+r) + D;
    Y3 = A(1+r-r:hh-r-r,1+r  :ww-r  ) + D;
    Y4 = A(1+r-r:hh-r-r,1+r-r:ww-r-r) + D;
    
    F(:,1) = hist(Y1(:), 1:(256*256));
    F(:,2) = hist(Y2(:), 1:(256*256));
    F(:,3) = hist(Y3(:), 1:(256*256));
    F(:,4) = hist(Y4(:), 1:(256*256));
    F1 = F(:);
    F0 = hist(A, 1:256);
    
elseif config == 1
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    %     X=double(X>=0);
    X=double(X>0);
    
    A=reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s)+1;
    [hh ww] = size(A);
    D  = (A(1+r  :hh-r  ,1+r  :ww-r) - 1) * 16;
    Y1 = A(1+r  :hh-r  ,1+r+r:ww-r+r) + D;
    Y2 = A(1+r-r:hh-r-r,1+r+r:ww-r+r) + D;
    Y3 = A(1+r-r:hh-r-r,1+r  :ww-r  ) + D;
    Y4 = A(1+r-r:hh-r-r,1+r-r:ww-r-r) + D;
    
    M = C(1+r:hh-r, 1+r:ww-r) > th;
    M = M(:);
    
    F(:,1) = hist(Y1(M), 1:(16*16));
    F(:,2) = hist(Y2(M), 1:(16*16));
    F(:,3) = hist(Y3(M), 1:(16*16));
    F(:,4) = hist(Y4(M), 1:(16*16));
    F1 = F(:);
    F0 = hist(A, 1:16);
    
    
elseif config == 2
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    X(2,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    X(3,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
    %     X=double(X>=0);
    X=double(X>0);
    
    A=reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s)+1;
%     B = zeros([16,size(A)]);
%     B((0:numel(A)-1)'*16+A(:)) = 1;
%     
%     [tt hh ww ] = size(B);
%     D  = B(:,1+r  :hh-r  ,1+r  :ww-r);
%     Y1 = B(:,1+r  :hh-r  ,1+r+r:ww-r+r);
%     Y2 = B(:,1+r-r:hh-r-r,1+r+r:ww-r+r);
%     Y3 = B(:,1+r-r:hh-r-r,1+r  :ww-r  );
%     Y4 = B(:,1+r-r:hh-r-r,1+r-r:ww-r-r);
%     
%     F = zeros(size(D,1), size(D,1), 4);
%     D=D(:,:);
%     F(:,:,1) = Y1(:,:)*D';
%     F(:,:,2) = Y2(:,:)*D';
%     F(:,:,3) = Y3(:,:)*D';
%     F(:,:,4) = Y4(:,:)*D';
%     
%     F0 = sum(B(:,:),2);
%     F1 = F(:);
%     
    [hh ww] = size(A);
    D  = (A(1+r  :hh-r  ,1+r  :ww-r) - 1) * 16;
    Y1 = A(1+r  :hh-r  ,1+r+r:ww-r+r) + D;
    Y2 = A(1+r-r:hh-r-r,1+r+r:ww-r+r) + D;
    Y3 = A(1+r-r:hh-r-r,1+r  :ww-r  ) + D;
    Y4 = A(1+r-r:hh-r-r,1+r-r:ww-r-r) + D;
    
    F(:,1) = hist(Y1(:), 1:(16*16));
    F(:,2) = hist(Y2(:), 1:(16*16));
    F(:,3) = hist(Y3(:), 1:(16*16));
    F(:,4) = hist(Y4(:), 1:(16*16));
    F1 = F(:);
    F0 = hist(A, 1:16);
  
end



