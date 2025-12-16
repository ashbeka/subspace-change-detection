function [MAP MASK1 MASK2] = cvtCyclicCircleSepFilter(Vimg, r, wi, wo)

L = 2 * (r + wo) + 1;
c = r+wo+1;
MASK1 = false(L,L);
MASK2 = false(L,L);
% MASK1 = false(L,L);
% MASK2 = false(L,L);
% List1 = zeros(L^2, 2);
% List2 = zeros(L^2, 2);
% N1 = 0;
% N2 = 0;
for px = 1:L;
    for py=1:L;
        if (r^2) >= ((c-py)^2 + (c-px)^2) && (r-wi)^2 <= ((c-py)^2 + (c-px)^2)
            %             N1 = N1 + 1;
            MASK1(py,px) = true;
            %             List1(N1,:) =[py,px];
        elseif (r+wo)^2 >= ((c-py)^2 + (c-px)^2) && (r^2) <= ((c-py)^2 + (c-px)^2)
            %             N2 = N2 + 1;
            MASK2(py,px) = true;
            %             List2(N2,:) =[py,px];
        end
    end
end
% List1 = List1(1:N1,:);
% List2 = List2(1:N2,:);


% foo = @(inp_hue) cvtCyclicSepFilter_gauss(inp_hue(MASK1), inp_hue(MASK2));
% foo = @(inp_hue) cvtCyclicSepFilter2(inp_hue(MASK1), inp_hue(MASK2));
% MAP = nlfilter(hue, [L L], foo);

imgH = size(Vimg, 1);
imgW = size(Vimg, 2);
MAP = zeros(imgH, imgW);
% V1cos = zeros(N1,1);
% V1sin = zeros(N1,1);
% V2cos = zeros(N2,1);
% V2sin = zeros(N2,1);

for wi = 1:imgW-L+1;
    for hi = 1:imgH-L+1;
        Vcos = Vimg(hi:(hi+L-1), wi:(wi+L-1), 1);
        Vsin = Vimg(hi:(hi+L-1), wi:(wi+L-1), 2);
        V1cos = Vcos(MASK1);
        V1sin = Vsin(MASK1);
        V2cos = Vcos(MASK2);
        V2sin = Vsin(MASK2);
        %         for l = 1:N1
        %             V1cos(l) = Vimg(List1(l,1)+hi-1,List1(l,2)+wi-1,1);
        %             V1sin(l) = Vimg(List1(l,1)+hi-1,List1(l,2)+wi-1,2);
        %         end
        %         for l = 1:N2
        %             V2cos(l) = Vimg(List2(l,1)+hi-1,List2(l,2)+wi-1,1);
        %             V2sin(l) = Vimg(List2(l,1)+hi-1,List2(l,2)+wi-1,2);
        %         end
        
        MAP(hi+c-1, wi+c-1) = cvtCyclicSepFilter([V1cos(:),V1sin(:)], [V2cos(:), V2sin(:)]);
    end
end

end
