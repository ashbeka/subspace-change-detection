function Z = cvtCOMB3(X1,X2,X3)
[nDim,nNum1] = size(X1);
[nDim,nNum2] = size(X2);
[nDim,nNum3] = size(X3);

Z = zeros(nDim,nNum1,nNum2,nNum3);
for I1 = 1:(nNum1)
for I2 = 1:(nNum2)
for I3 = 1:(nNum3)
        Z(:,I1,I2,I3) = X1(:,I1) + X2(:,I2) + X3(:,I3);
end
end
end

Z = Z(:,:);

