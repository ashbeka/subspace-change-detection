function [F1 F0] = cvtLBPLAC_C(img, s, r, num_block_h, num_block_w, option)
% ï‚êîï\åªÇÃLBPLAC
%

if nargin < 4
    num_block_h = 1;
    num_block_w = 1;
end

if exist('option', 'var')
    
    Z = double(img);
    [h w] = size(Z);
    
    C = Z(1+s:h-s,1+s:w-s);
    Xtmp = zeros(4,h-2*s,w-2*s);
    Xtmp(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    Xtmp(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    Xtmp(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    Xtmp(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    
    X = zeros(2,4,h-2*s,w-2*s);
    X(1,:,:,:) = double(Xtmp>0);
    X(2,:,:,:) = double(Xtmp<=0);
    X = reshape(X, 4*2, h-2*s, w-2*s);
    [tt hh ww] = size(X);
    
    D  = X(:,1+r:hh-r,1+r:ww-r);
    Y1 = X(:,1+r  :hh-r  ,1+r+r:ww-r+r);
    Y2 = X(:,1+r-r:hh-r-r,1+r+r:ww-r+r);
    Y3 = X(:,1+r-r:hh-r-r,1+r  :ww-r  );
    Y4 = X(:,1+r-r:hh-r-r,1+r-r:ww-r-r);
    
    [tt hhh www] = size(D);
    hhh = floor(hhh/num_block_h)*num_block_h;
    www = floor(www/num_block_w)*num_block_w;
    size_block_h = hhh / num_block_h;
    size_block_w = www / num_block_w;
    
    F1 = zeros(tt, tt, 4, num_block_h, num_block_w);
    F0 = zeros(tt, num_block_h, num_block_w);
    
    for nhi = 1:num_block_h
        ylist = (1:size_block_h)+(nhi-1)*size_block_h;
        for nwi = 1:num_block_w
            xlist = (1:size_block_w)+(nwi-1)*size_block_w;
            
            Dtmp  =  D(:, ylist, xlist);
            Y1tmp = Y1(:, ylist, xlist);
            Y2tmp = Y2(:, ylist, xlist);
            Y3tmp = Y3(:, ylist, xlist);
            Y4tmp = Y4(:, ylist, xlist);
            
            Dtmp = Dtmp(:,:);
            F1(:,:,1,nhi,nwi) = Y1tmp(:,:)*Dtmp';
            F1(:,:,2,nhi,nwi) = Y2tmp(:,:)*Dtmp';
            F1(:,:,3,nhi,nwi) = Y3tmp(:,:)*Dtmp';
            F1(:,:,4,nhi,nwi) = Y4tmp(:,:)*Dtmp';
            
            Xtmp = X(:,ylist+r, xlist+r);
            F0(:,nhi,nwi) = sum(Xtmp(:,:), 2);
        end
    end
    
    F1 = F1(:);
    F0 = F0(:);
else
    
    Z = double(img);
    [h w] = size(Z);
    
    C = Z(1+s:h-s,1+s:w-s);
    Xtmp = zeros(8,h-2*s,w-2*s);
    Xtmp(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    Xtmp(2,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    Xtmp(3,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    Xtmp(4,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    Xtmp(5,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    Xtmp(6,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    Xtmp(7,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    Xtmp(8,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
    
    X = zeros(2,8,h-2*s,w-2*s);
    X(1,:,:,:) = double(Xtmp>0);
    X(2,:,:,:) = double(Xtmp<=0);
    X = reshape(X, 8*2, h-2*s, w-2*s);
    [tt hh ww] = size(X);
    
    D  = X(:,1+r:hh-r,1+r:ww-r);
    Y1 = X(:,1+r  :hh-r  ,1+r+r:ww-r+r);
    Y2 = X(:,1+r-r:hh-r-r,1+r+r:ww-r+r);
    Y3 = X(:,1+r-r:hh-r-r,1+r  :ww-r  );
    Y4 = X(:,1+r-r:hh-r-r,1+r-r:ww-r-r);
    
    [tt hhh www] = size(D);
    hhh = floor(hhh/num_block_h)*num_block_h;
    www = floor(www/num_block_w)*num_block_w;
    size_block_h = hhh / num_block_h;
    size_block_w = www / num_block_w;
    
    F1 = zeros(tt, tt, 4, num_block_h, num_block_w);
    F0 = zeros(tt, num_block_h, num_block_w);
    
    for nhi = 1:num_block_h
        ylist = (1:size_block_h)+(nhi-1)*size_block_h;
        for nwi = 1:num_block_w
            xlist = (1:size_block_w)+(nwi-1)*size_block_w;
            
            Dtmp  =  D(:, ylist, xlist);
            Y1tmp = Y1(:, ylist, xlist);
            Y2tmp = Y2(:, ylist, xlist);
            Y3tmp = Y3(:, ylist, xlist);
            Y4tmp = Y4(:, ylist, xlist);
            
            Dtmp = Dtmp(:,:);
            F1(:,:,1,nhi,nwi) = Y1tmp(:,:)*Dtmp';
            F1(:,:,2,nhi,nwi) = Y2tmp(:,:)*Dtmp';
            F1(:,:,3,nhi,nwi) = Y3tmp(:,:)*Dtmp';
            F1(:,:,4,nhi,nwi) = Y4tmp(:,:)*Dtmp';
            
            Xtmp = X(:,ylist+r, xlist+r);
            F0(:,nhi,nwi) = sum(Xtmp(:,:), 2);
        end
    end
    
    F1 = F1(:);
    F0 = F0(:);
    
end
% function [F1 F0] = cvtLBPLAC_C(img,s,r)
% % ï‚êîï\åªÇÃLBPLAC
% %
% %
%
% Z = double(img);
% [h w] = size(Z);
%
% C = Z(1+s:h-s,1+s:w-s);
% Xtmp = zeros(8,h-2*s,w-2*s);
% Xtmp(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
% Xtmp(2,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
% Xtmp(3,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
% Xtmp(4,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
% Xtmp(5,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
% Xtmp(6,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
% Xtmp(7,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
% Xtmp(8,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
%
% X = zeros(2,8,h-2*s,w-2*s);
% X(1,:,:,:) = double(Xtmp>0);
% X(2,:,:,:) = double(Xtmp<=0);
% X = reshape(X, 8*2, h-2*s, w-2*s);
% [tt hh ww] = size(X);
% D  = X(:,1+r:hh-r,1+r:ww-r);
% Y1 = X(:,1+r  :hh-r  ,1+r+r:ww-r+r);
% Y2 = X(:,1+r-r:hh-r-r,1+r+r:ww-r+r);
% Y3 = X(:,1+r-r:hh-r-r,1+r  :ww-r  );
% Y4 = X(:,1+r-r:hh-r-r,1+r-r:ww-r-r);
%
% F = zeros(tt, tt, 4);
%
% D=D(:,:);
% F(:,:,1) = Y1(:,:)*D';
% F(:,:,2) = Y2(:,:)*D';
% F(:,:,3) = Y3(:,:)*D';
% F(:,:,4) = Y4(:,:)*D';
%
% F0 = sum(X(:,:),2);
% F1 = F(:);
