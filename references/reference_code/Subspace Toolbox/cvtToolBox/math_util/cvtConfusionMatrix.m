function C = cvtConfusionMatrix(SIM, L)

nClass = size(unique(L),2);

C = zeros(nClass,nClass);
for I=1:size(SIM,2);
    [val ind] = max(SIM(:,I));
    C(L(I),ind) = C(L(I),ind) + 1;     
    %disp("----")
end

