function f = cvtDBPP(img, model)
    c1 = model.config1;
    c2 = model.config2;
    smax = max([c1,c2],[],2);
    smin = min([c1,c2],[],2);
    nc = size(c1, 2);
    [h w] = size(img);
    B = zeros(nc, h-(smax(1)-smin(1)), w-(smax(2)-smin(2)));
    for ic = 1:nc
        tmp = img(1-smin(1)+c1(1,ic):end-smax(1)+c1(1,ic),1-smin(2)+c1(2,ic):end-smax(2)+c1(2,ic)) ...
              > img(1-smin(1)+c2(1,ic):end-smax(1)+c2(1,ic),1-smin(2)+c2(2,ic):end-smax(2)+c2(2,ic));
        B(ic,:,:) = double(tmp);
    end
    B = B(:,:);
    indexes = (2.^(0:nc-1))*B(:,:);
    f = hist(indexes, 0:(2^nc)-1); 
end
