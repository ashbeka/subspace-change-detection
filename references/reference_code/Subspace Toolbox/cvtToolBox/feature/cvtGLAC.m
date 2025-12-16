function [f ff] = cvtGLAC(img, r)
% [f ff] = cvtGLAC(img, r)
% ++ input ++
%   img : gray image
%   r : reference step
% ++ output ++
%  f : GLAC features (256dim)
%  ff: GLAC features at each pixels (256xHxW)
%
% 

bin_num = 8;

if ~isa(img, 'double')
    img = im2double(img);
end

[h w] = size(img);
[grad_img_bin, grad_img_norm] = cvtIm2GradientVecImgInBin(img, bin_num);

B1 = grad_img_bin(:, 1  :h-r-r, 1    :w-r-r); B1 = B1(:,:);
B2 = grad_img_bin(:, 1  :h-r-r, 1+r  :w-r);   B2 = B2(:,:);
B3 = grad_img_bin(:, 1  :h-r-r, 1+r+r:w);     B3 = B3(:,:);
B4 = grad_img_bin(:, 1+r:h-r,   1    :w-r-r); B4 = B4(:,:);
B5 = grad_img_bin(:, 1+r:h-r,   1+r  :w-r);   B5 = B5(:,:);

N1 = grad_img_norm(1:h-r-r, 1:w-r-r); N1 = N1(:)';
N2 = grad_img_norm(1:h-r-r, 1+r:w-r); N2 = N2(:)';
N3 = grad_img_norm(1:h-r-r, 1+r+r:w); N3 = N3(:)';
N4 = grad_img_norm(1+r:h-r, 1:w-r-r); N4 = N4(:)';
N5 = grad_img_norm(1+r:h-r, 1+r:w-r); N5 = N5(:)';

if nargout < 2
    f = zeros(bin_num, bin_num, 4);
    f(:,:,1) = repmat(min(N5,N1),bin_num,1).*B1*B5';
    f(:,:,2) = repmat(min(N5,N2),bin_num,1).*B2*B5';
    f(:,:,3) = repmat(min(N5,N3),bin_num,1).*B3*B5';
    f(:,:,4) = repmat(min(N5,N4),bin_num,1).*B4*B5';  
    f = f(:);

elseif nargout >= 2
    BB1 = repmat(min(N5,N1),bin_num,1).*B1;
    BB2 = repmat(min(N5,N2),bin_num,1).*B2;
    BB3 = repmat(min(N5,N3),bin_num,1).*B3;
    BB4 = repmat(min(N5,N4),bin_num,1).*B4;

    ff = zeros((w-2*r)*(h-2*r), bin_num, bin_num,  4);
    for ib = 1:bin_num
        bb = repmat(B5(ib,:),bin_num,1);
        ff(:,:,ib,1) = (bb.*BB1)';
        ff(:,:,ib,2) = (bb.*BB2)';
        ff(:,:,ib,3) = (bb.*BB3)';
        ff(:,:,ib,4) = (bb.*BB4)';
    end
    f = sum(ff,1);
    f = f(:);

    ff = permute(ff, [2 3 4 1]);
    ff = reshape(ff, bin_num*bin_num*4, (h-2*r), (w-2*r));
end

end

function [grad_img_bin, grad_img_norm] = cvtIm2GradientVecImgInBin(img, bin_num)

weight_num = 2;

[h,w] = size(img);

H1 = [1,0;0,-1];
H2 = fliplr(H1);
grad_img(1,:,:) = imfilter(img, H1, 'replicate');
grad_img(2,:,:) = imfilter(img, H2, 'replicate');
% grad_img(1,:,:) = imfilter(img, [-1,0,1;-2,0,2;-1,0,1]);
% grad_img(2,:,:) = imfilter(img, [1,2,1;0,0,0;-1,-2,-1]);
grad_img = grad_img(:,:);

grad_img_norm = sqrt(grad_img(1,:).^2 + grad_img(2,:).^2);
grad_img(grad_img_norm ~= 0) = grad_img(grad_img_norm ~= 0)...
                                ./grad_img_norm(grad_img_norm ~= 0);

ths = (1:bin_num) * 2*pi / bin_num;
standard_ori =[cos(ths); sin(ths)];

grad_img_bin = zeros(bin_num, h, w);
dist = cvtL2Norm(standard_ori, grad_img);
[dist_sort dist_sort_index] = sort(dist, 1, 'ascend');

weight = zeros(bin_num, h*w);
weight((dist_sort_index(1,:)+((0:h*w-1)*bin_num))) = 1;
for iw = 2:weight_num
    weight(dist_sort_index(iw,:)+(0:h*w-1)*bin_num) ...
        = dist_sort(1,:) ./ dist_sort(iw,:);
end

sum_weight_inv = 1./sum(weight, 1);
for iw = 1:bin_num
    grad_img_bin(iw,:) = weight(iw,:) .* sum_weight_inv;
end

grad_img_bin = reshape(grad_img_bin, bin_num, h,w);
grad_img_norm = reshape(grad_img_norm, h,w);

end
