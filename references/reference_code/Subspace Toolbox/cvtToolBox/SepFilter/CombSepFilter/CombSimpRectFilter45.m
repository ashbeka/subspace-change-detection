function MAP = CombSimpRectFilter45(I,P,sh)


r = round(sh/sqrt(2));
w = ceil(sh/3/sqrt(2));
br = 2*r;

% I = IntegralImage45(X);
% P = IntegralImage45(X.^2);
% ìÒèÊêœï™ÉCÉÅÅ[ÉW

MAP(:,:,1) = tmpFnc(I,P,r,br,w,w);
MAP(:,:,2) = tmpFnc(I,P,w,w,r,br);


end

%%
function MAP = tmpFnc(I,P,r,br,w,bw)

MAP = zeros(size(I)-1);
[H W] = size(MAP);

h = bw+br+2;
N =(2*bw+1)*(2*(1+2*br));
N1=(2*w+1)*(2*(1+2*r));
N2=N-N1;

HH = bw + br +1+1:H-(bw + br +2);
WW = bw + br +3:W-(bw + br+1);
P1 = I(HH - bw - br -1      ,WW + bw - br - 1);
P2 = I(HH + bw - br -1+1    ,WW - bw - br - 1-1  );
P3 = I(HH + bw + br +1      ,WW - bw + br - 1);
P4 = I(HH - bw + br         ,WW + bw + br    );
S = (P4+P2-P3-P1);

P1 = P(HH - bw - br -1      ,WW + bw - br - 1);
P2 = P(HH + bw - br -1+1    ,WW - bw - br - 1-1  );
P3 = P(HH + bw + br +1      ,WW - bw + br - 1);
P4 = P(HH - bw + br         ,WW + bw + br    );
T= (P4+P2-P3-P1);

M = S./N;
Y = T./N;
St = Y - M.^2;

P1 = I(HH - w - r -1      ,WW + w - r - 1);
P2 = I(HH + w - r -1+1    ,WW - w - r - 1-1  );
P3 = I(HH + w + r +1      ,WW - w + r - 1);
P4 = I(HH - w + r         ,WW + w + r    );
S1= (P4+P2-P3-P1);

S2=S-S1;
M1=S1/N1;
M2=S2/N2;

Sb = ((N1*((M1-M).^2)) + (N2*((M2-M).^2)))/N;

MAP(h:end-h,h:end-h)=(Sb./St).*sign(M2-M1);
MAP(isnan(MAP))=0;
MAP(isinf(MAP))=0;
end

