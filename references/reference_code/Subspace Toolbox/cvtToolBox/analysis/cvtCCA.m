function [Y1, Y2, A1, A2, M1, M2, D] = cvtCCA(X1, X2, k, alpha)
% ³€‘ŠŠÖ•ªÍ@[Canonical Correlation Analys]
% Input
% X1:InputDim x DataNum matrix
% X2:InputDim x DataNum matrix
% Output
% Y1:mapped X1
% Y2:mapped X2
% A1:map matrix X1
% A2:map matrix X2
% M1:mean X1
% M2:mean X2
% D :canonical angle cos^2 ???
% 
% Written by nosaka at 2010/10/07

if(nargin < 4)
    alpha = 0;
end

%     r = min(size(X1,1), size(X2,1)); % ŸŒ³‚ª¬‚³‚¢•û
n = size(X1, 2);
% %
[X1, M1] = cvtCentering(X1);
[X2, M2] = cvtCentering(X2);
% M1 = zeros(size(X1,1), 1);
% M2 = zeros(size(X2,1), 1);

%
S11 = X1*X1'/n;
S22 = X2*X2'/n;
S12 = X1*X2'/n;
% 
% B = [S11, zeros(size(S11,1),size(S22,1)); ...
%     zeros(size(S22,1),size(S11,1)), S22];
% 
% Binv = pinv(B);
% 
% if(~isnumeric(alpha))
%    alpha = det(Binv); 
% end

S11 = S11 + alpha * eye(size(S11));
S22 = S22 + alpha * eye(size(S22));

% éë‚Ì“m‚ğ‚à‚Æ‚Éì‚Á‚½(ŠÔˆá‚Á‚Ä‚éL‚¢)
% [A1, D, A2] = svd(S11^(-1/2)*S12*S22^(-1/2));

% Wikipedia base
% [A1 D] = eigs(S12*S22^(-1)*S12', S11);  % S12*S22^(-1)*S12' x = ƒÉ S11 x
[A1 D] = eigs(S12/S22*S12', S11, k);  % S12*S22^(-1)*S12' x = ƒÉ S11 x
A1 = real(A1);
A2 = S22\S12'*A1;

D = diag(D);
Y1 = A1'*X1;
Y2 = A2'*X2;



end
