function [f ff] = cvtCILAC(img, r)
% cvtCILAC
% ++ input ++
%   img : rgb image
%   r : reference step
% ++ output ++
%  f : CILAC features (900dim)
%

bin_num = 15;

if ~isa(img, 'double')
    img = im2double(img);
end

[h,w,c] = size(img);
ci_bin = cvtIm2ColorIndexVecImg(img);
f = zeros(bin_num, bin_num, 4);

B1 = ci_bin(:, 1  :h-r-r, 1    :w-r-r); B1 = B1(:,:);
B2 = ci_bin(:, 1  :h-r-r, 1+r  :w-r);   B2 = B2(:,:);
B3 = ci_bin(:, 1  :h-r-r, 1+r+r:w);     B3 = B3(:,:);
B4 = ci_bin(:, 1+r:h-r,   1    :w-r-r); B4 = B4(:,:);
B5 = ci_bin(:, 1+r:h-r,   1+r  :w-r);   B5 = B5(:,:);

if nargout < 2
    f = zeros(bin_num, bin_num, 4);
    f(:,:,1) = B1*B5';
    f(:,:,2) = B2*B5';
    f(:,:,3) = B3*B5';
    f(:,:,4) = B4*B5';  
    f = f(:);

elseif nargout >= 2
    ff = zeros((w-2*r)*(h-2*r), bin_num, bin_num,  4);
    for ib = 1:bin_num
        bb = repmat(B5(ib,:),bin_num,1);
        ff(:,:,ib,1) = (bb.*B1)';
        ff(:,:,ib,2) = (bb.*B2)';
        ff(:,:,ib,3) = (bb.*B3)';
        ff(:,:,ib,4) = (bb.*B4)';
    end
    f = sum(ff,1);
    f = f(:);

    ff = permute(ff, [2 3 4 1]);
    ff = reshape(ff, bin_num*bin_num*4, (h-2*r), (w-2*r));
end

%
% % 1st order
% f(:,:,1) = B5*B1';
% f(:,:,2) = B5*B2';
% f(:,:,3) = B5*B3';
% f(:,:,4) = B5*B4';
% f = f(:);
% 
% if nargout >= 2
%     ff = zeros(bin_num, bin_num, (w-2*r)*(h-2*r), 4);
%     for ib = 1:bin_num
%         bb = repmat(B5(ib,:),bin_num,1);
%         ff(ib,:,:,1) = bb.*B1;
%         ff(ib,:,:,2) = bb.*B2;
%         ff(ib,:,:,3) = bb.*B3;
%         ff(ib,:,:,4) = bb.*B4;
%     end
%     ff = permute(ff, [1 2 4 3]);
%     ff = reshape(ff, bin_num*bin_num*4, (h-2*r), (w-2*r));
% end

end

function [ci_bin] = cvtIm2ColorIndexVecImg(img)

cir_num = 6;
bin_num = 3 + 2*cir_num;
weight_num = 4;

ths = (1:cir_num) / cir_num * 2 * pi;
standard_ori =[[1;0;0],[0.5;0;0],[0;0;0],...
    [0.5*ones(1,cir_num);0.5*cos(ths); 0.5*sin(ths);],...
    [    ones(1,cir_num);    cos(ths);     sin(ths);],...
    ];

img = rgb2hsv(img);
[h,w,c] = size(img);
img = permute(img, [3 1 2]);
img = img(:,:);

c_img = [img(3,:);...
        img(2,:).*img(3,:).*cos(img(1,:)*2*pi);...
        img(2,:).*img(3,:).*sin(img(1,:)*2*pi)];

dist = cvtL2Norm(standard_ori, c_img);
[dist_sort dist_sort_index] = sort(dist, 1, 'ascend');

weight = zeros(bin_num, h*w);
weight((dist_sort_index(1,:)+((0:h*w-1)*bin_num))) = 1;
for iw = 2:weight_num
    weight(dist_sort_index(iw,:)+(0:h*w-1)*bin_num) ...
        = dist_sort(1,:) ./ dist_sort(iw,:);
end

ci_bin = repmat(1./sum(weight, 1), bin_num,1) .* weight;
ci_bin = reshape(ci_bin, bin_num, h,w);

end
