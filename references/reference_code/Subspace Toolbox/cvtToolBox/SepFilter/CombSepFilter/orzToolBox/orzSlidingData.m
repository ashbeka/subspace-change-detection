function Y = orzSlidingData(X,nSubNum,nSlide)

nDim = size(X,1);
nNum = size(X,2);
nSizeX = size(X);
nSet = floor((nNum-nSubNum)/nSlide)+1;
Y = zeros([nDim,nSubNum,nSet,nSizeX(3:end)]);

for I=1:nSet
    Y( :,:,I,:) = X(:,1 + (I-1)*nSlide: nSubNum + (I-1)*nSlide,:);
end