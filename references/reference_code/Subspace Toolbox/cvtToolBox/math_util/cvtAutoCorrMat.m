function R = cvtAutoCorrMat(X)
% cvtAutoCorrMat
%  ©ŒÈ‘ŠŠÖs—ñ‚ğŒvZ‚·‚é

R = X(:,:)*X(:,:)'/size(X(:,:),2);
end
