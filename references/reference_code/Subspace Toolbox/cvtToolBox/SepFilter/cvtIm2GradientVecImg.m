function grad_img = cvtIm2GradientVecImg(gray_img)

if(ndims(gray_img) ~= 2)
    error('input image is not gray scale image.');
end

gray_img = im2double(gray_img);

Vcos = imfilter(gray_img, [-1,0,1;-2,0,2;-1,0,1]);
Vsin = imfilter(gray_img, [1,2,1;0,0,0;-1,-2,-1]);
% V = cvtNormalize([Vcos(:), Vsin(:)]');
% Vcos = V(1,:) .* gray_img(:)';
% Vsin = V(2,:) .* gray_img(:)';
% Vcos = reshape(Vcos, size(gray_img));
% Vsin = reshape(Vsin, size(gray_img));

grad_img = zeros([size(gray_img) 2]);
grad_img(:,:,1) = Vcos;
grad_img(:,:,2) = Vsin;

end
