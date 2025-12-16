function Z = cvtCOMB2(X1,X2)
[nDim,nNum1] = size(X1);
[nDim,nNum2] = size(X2);
Z = zeros(nDim,nNum1,nNum2);
for I1 = 1:(nNum1)
    for I2 = 1:(nNum2)
        Z(:,I1,I2) = X1(:,I1) + X2(:,I2);
    end
end

Z = Z(:,:);

