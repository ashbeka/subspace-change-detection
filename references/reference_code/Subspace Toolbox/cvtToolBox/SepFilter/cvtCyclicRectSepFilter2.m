function [MAP] = cvtCyclicRectSepFilter2(Vimg, h, w1, w2, tate_flag)
%
% +--w2--+-w1+  +w1+---w2---+ 
% Å°Å°Å°Å°Å†Å†Å†Å†Å†Å°Å°Å°Å°  |
% Å°R2Å°Å°Å†Å†R1Å†Å†Å°Å°R2Å°  h
% Å°Å°Å°Å°Å†Å†Å†Å†Å†Å°Å°Å°Å°  |
% Å°Å°Å°Å°Å†Å†ÇwÅ†Å†Å°Å°Å°Å°Å@
% Å°Å°Å°Å°Å†Å†Å†Å†Å†Å°Å°Å°Å°  |
% Å°Å°Å°Å°Å†Å†Å†Å†Å†Å°Å°Å°Å°  h
% Å°Å°Å°Å°Å†Å†Å†Å†Å†Å°Å°Å°Å°  |
%
%

imgH = size(Vimg, 1);
imgW = size(Vimg, 2);
MAP = zeros(imgH, imgW);

if(tate_flag == false)
    for wi = (w2+w1+1):(imgW-w1-w2);
        for hi = (h+1):(imgH-h);
            V1cos = Vimg((hi-h):(hi+h), (wi-w1):(wi+w1), 1);
            V1sin = Vimg((hi-h):(hi+h), (wi-w1):(wi+w1), 2);
            V21cos = Vimg((hi-h):(hi+h), (wi-w1-w2):(wi-w1-1), 1);
            V21sin = Vimg((hi-h):(hi+h), (wi-w1-w2):(wi-w1-1), 2);
            V22cos = Vimg((hi-h):(hi+h), (wi+w1+1):(wi+w1+w2), 1);
            V22sin = Vimg((hi-h):(hi+h), (wi+w1+1):(wi+w1+w2), 2);
            
           MAP(hi, wi) = cvtCyclicSepFilter([V1cos(:),V1sin(:)], [V21cos(:), V21sin(:);V22cos(:), V22sin(:)]);
        end
    end    
else
    for hi = (w2+w1+1):(imgH-w1-w2);
        for wi = (h+1):(imgW-h);
            V1cos = Vimg((hi-w1):(hi+w1), (wi-h):(wi+h), 1);
            V1sin = Vimg((hi-w1):(hi+w1), (wi-h):(wi+h), 2);
            V21cos = Vimg((hi-w1-w2):(hi-w1-1), (wi-h):(wi+h), 1);
            V21sin = Vimg((hi-w1-w2):(hi-w1-1), (wi-h):(wi+h), 2);
            V22cos = Vimg((hi+w1+1):(hi+w1+w2), (wi-h):(wi+h), 1);
            V22sin = Vimg((hi+w1+1):(hi+w1+w2), (wi-h):(wi+h), 2);
            MAP(hi, wi) = cvtCyclicSepFilter([V1cos(:),V1sin(:)], [V21cos(:), V21sin(:);V22cos(:), V22sin(:)]);
        end
    end

end


end
