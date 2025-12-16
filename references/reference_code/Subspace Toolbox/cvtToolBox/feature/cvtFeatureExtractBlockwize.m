function f = cvtFeatureExtractBlockwize(func_feature, dim_feature, img,  num_h, num_w)

s = size(img);

size_h = floor(s(1) / num_h);
size_w = floor(s(2) / num_w);

img = img(1:size_h*num_h, 1:size_w*num_w, :);

f = zeros(dim_feature, num_h, num_w);

for wi = 1:num_w
for hi = 1:num_h
    f(:, hi, wi) = func_feature(img(((hi-1)*size_h+1):(hi*size_h),((wi-1)*size_w+1):(wi*size_w), :));    
end
end

f = f(:);

end
