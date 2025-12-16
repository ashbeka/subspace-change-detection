function PEAK_MAP = cvtLocalPeak(MAP)

[H W] = size(MAP);
PEAK_MAP = zeros(H, W);

for iy = 2:(H-1)
for ix = 2:(W-1)

    n = [MAP(iy-1,ix+(-1:1)), MAP(iy,ix+(-1:1)), MAP(iy,ix+(-1:1))];
    [~, index] = max(n);
    if index(1) == 5
        PEAK_MAP(iy,ix) = 1;
    end
end
end

end
% 
% function PEAK_MAP = cvtPeak(MAP)
% 
% H = size(MAP, 1);
% W = size(MAP, 2);
% PEAK_MAP = zeros(H, W);
% 
% for iy = 1:H
% for ix = 1:W
% 
%     [~, index] = max(MAP(iy,ix, :));
%     if ((index ~= 1) || (index ~= size(index,1)))
%         PEAK_MAP(iy,ix) = 1;
%     end
% end
% end
% 
% end
