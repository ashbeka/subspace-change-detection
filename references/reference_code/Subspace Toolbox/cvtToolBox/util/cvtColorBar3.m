function cvtColorBar3(Z)


H = bar3(Z);
for I = 1:length(H)
    zdata = repmat(cvtVec(repmat(Z(:,I)',6,1)),1,4); 
    set(H(I),'Cdata',zdata);
end
colormap jet
colorbar
