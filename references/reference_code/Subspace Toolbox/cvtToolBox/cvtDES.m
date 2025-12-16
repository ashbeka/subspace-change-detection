function V = cvtDES(X)

SizeX = size(X);
if SizeX(1) ~= 64
    X = imresize(X,64/SizeX(1));
end

X = bwmorph(im2bw(X),'remove');
nSize = 64;
nSizeBlock = 8;

V = zeros(4,7,7);
B = X(1+1 :end-1  ,1+1:end-1);
B1 = X(1+1:end-1  ,1+2:end);
B2 = X(1  :end-2  ,1+2:end);
B3 = X(1  :end-2  ,1+1:end-1);
B4 = X(1  :end-2  ,1  :end-2);

Z = zeros(nSize,nSize,4);
Z(2:end-1,2:end-1,1) =  B.*B1;
Z(2:end-1,2:end-1,2) =  B.*B2;
Z(2:end-1,2:end-1,3) =  B.*B3;
Z(2:end-1,2:end-1,4) =  B.*B4;



for I=1:7
    for J=1:7
        tmpZ = Z(1 + (I-1)*nSizeBlock:(I+1)*nSizeBlock,1 + (J-1)*nSizeBlock:(J+1)*nSizeBlock,:);
        V(:,I,J) = cvtVec(sum(sum(tmpZ,1),2));       
    end
end
    
V = V(:);

