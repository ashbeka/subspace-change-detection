function img = cvtMontage(imgs, hm, wm)
% function img = cvtMontage(imgs, h, w)
% 
% [input]
% imgs : input gray or rgb images
% hm, wm : number of horizontal and vartical images
% 
% [output]
% img : montage image

% rgb image
if ndims(imgs) >= 4
    imgs = imgs(:,:,:,:);
    [h w c n] = size(imgs);
    img = reshape(permute(reshape(imgs, [h,w,c,hm,wm]), [1 4 2 5 3]), [h*hm, w*wm, c]);

% gray image
else
    imgs = imgs(:,:,:);
    [h w n] = size(imgs);
    img = reshape(permute(reshape(imgs, [h,w,hm,wm]), [1 3 2 4]), [h*hm, w*wm]);
end
    


end
