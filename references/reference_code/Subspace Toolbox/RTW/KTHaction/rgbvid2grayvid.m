function B = rgbvid2grayvid(A)

if ~isempty(A)
    sa = size(A);
    B = zeros(sa(1),sa(2),sa(4));
    
    for i = 1:sa(4)
        B(:,:,i) = rgb2gray(A(:,:,:,i));
    end
else
    B = [];
end
end