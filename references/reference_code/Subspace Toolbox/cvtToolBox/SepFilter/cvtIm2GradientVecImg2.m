function [grad_img_bin, grad_img_norm] = cvtIm2GradientVecImg2(img, bin_num)

weight_num = 2;

if ~isa(img, 'double')
   img = im2double(img); 
end

[h,w] = size(img);
grad_img(1,:,:) = imfilter(img, [-1,0,1;-2,0,2;-1,0,1]);
grad_img(2,:,:) = imfilter(img, [1,2,1;0,0,0;-1,-2,-1]);
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

sum_weight = sum(weight, 1);
for iw = 1:bin_num
    grad_img_bin(iw,:) = weight(iw,:) ./ sum_weight;
end

grad_img_bin = reshape(grad_img_bin, bin_num, h,w);
grad_img_norm = reshape(grad_img_norm, h,w);


% for wi = 1:w
% for hi = 1:h
%    th = atan2(grad_img(hi,wi,1), grad_img(hi,wi,2)) + pi;
%    thi1 = floor(th/dth);
%    thi2 = ceil(th/dth);
%    th2 = thi2 * dth;
%    tmp = (th2 - th) / dth;
%    if thi2 == bin_num+1
%       grad_img_bin(hi,wi,bin_num) = tmp; 
%       grad_img_bin(hi,wi,1) = 1 - tmp;
%    else
%        grad_img_bin(hi,wi,thi1+1) = tmp;
%        grad_img_bin(hi,wi,thi2+1) = 1 - tmp;
%    end
% end
% end


end
