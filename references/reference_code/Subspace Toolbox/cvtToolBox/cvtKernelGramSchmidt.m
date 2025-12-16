
function B = cvtKernelGramSchmidt(A,K)

B = zeros(size(A));
%% ÇÃÇÈÇﬁÇÃê≥ãKâª
B(:,1) = A(:,1)/sqrt((A(:,1)'*K*A(:,1)));
for I=2:size(A,2);
    v = A(:,I);
    for J=1:I-1;        
        v = v - (B(:,J)'*K*A(:,I))*B(:,J);
    end;
    B(:,I) = v/sqrt((v'*K*v));
end;
