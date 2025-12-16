function [F1 F0] = cvtLBPLAC(img, s, r, num_block_h, num_block_w, option)
%
% %% 5 78 151 224
%
if nargin < 4
    num_block_h = 1;
    num_block_w = 1;
end

% ŠÈ‘f”Å LBPLAC ã‰º¶‰E‚µ‚©Œ©‚È‚¢
if exist('option', 'var')
    Z = double(img);
    [h w] = size(Z);
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(4,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    X = double(X>0);
    
    [tt hh ww] = size(X);
    D  = X(:,1+r  :hh-r  ,1+r  :ww-r);
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
    
    F0 = F0(:);
    F1 = F1(:);
    
else
    
    Z = double(img);
    [h w] = size(Z);
    
    C = Z(1+s:h-s,1+s:w-s);
    X = zeros(8,h-2*s,w-2*s);
    X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
    X(2,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
    X(3,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
    X(4,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
    X(5,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
    X(6,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
    X(7,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
    X(8,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
    X = double(X>0);
    
    [tt hh ww] = size(X);
    D  = X(:,1+r  :hh-r  ,1+r  :ww-r);
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
    
    F0 = F0(:);
    F1 = F1(:);
    
end

%
% function [F1 F0] = cvtLBPLAC(img, s, r)
% Z = double(img);
% [h w] = size(Z);
%
% C = Z(1+s:h-s,1+s:w-s);
% X = zeros(8,h-2*s,w-2*s);
% X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s)-C;
% X(2,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s)-C;
% X(3,:,:) = Z(1+s-s:h-s-s,1+s  :w-s  )-C;
% X(4,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s)-C;
% X(5,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s)-C;
% X(6,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s)-C;
% X(7,:,:) = Z(1+s+s:h-s+s,1+s  :w-s  )-C;
% X(8,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s)-C;
% X=double(X>0);
%
% [tt hh ww] = size(X);
% D  = X(:,1+r  :hh-r  ,1+r  :ww-r);
% Y1 = X(:,1+r  :hh-r  ,1+r+r:ww-r+r);
% Y2 = X(:,1+r-r:hh-r-r,1+r+r:ww-r+r);
% Y3 = X(:,1+r-r:hh-r-r,1+r  :ww-r  );
% Y4 = X(:,1+r-r:hh-r-r,1+r-r:ww-r-r);
% F = zeros(8,8,4);
% D = D(:,:);
% F(:,:,1) = Y1(:,:)*D';
% F(:,:,2) = Y2(:,:)*D';
% F(:,:,3) = Y3(:,:)*D';
% F(:,:,4) = Y4(:,:)*D';
%
% F0 = sum(X(:,:),2);
% F1 = F(:);


% 1-37, 10-46+64, 19-55+64*2, 28-64+64*3
% 9-73 29-93 37-101 111-175 136-186 166-173
% function f = cvtLBPLAC(img, r)
% cvtLBPLAC
% input
%  img : gray scale image
%  r : step
% output
%  f : feature (264)

% [h w] = size(img);
%
% lbp_img = zeros(8, h-2, w-2);
%
% % calc Local Binary Pattern
% for wi = 2:w-1
% for hi = 2:h-1
%    local_img_vec = img([hi-1 hi hi+1], [wi-1 wi wi+1]);
%    local_img_vec = local_img_vec(:);
%    threshold = local_img_vec(5);
%    lbp_img(:,wi-1,hi-1) = double(local_img_vec([1 2 3 4 6 7 8 9]) > threshold);
% end
% end
%
% % calc LAC
% f0 = sum(lbp_img(:,:), 2);
% f1 = zeros(8*4, 8);
%
% for wi = 1+r:w-2-r
% for hi = 1+r:h-2-r
%     f1 = f1 + [lbp_img(:,hi-r,wi-r);lbp_img(:,hi-r,wi); ...
%                lbp_img(:,hi-r,wi+r);lbp_img(:,hi,wi-r)] ...
%                * lbp_img(:,hi,wi)';
% end
% end
%
% f = [cvtNormalize(f0); cvtNormalize(f1(:))];
%
% end
