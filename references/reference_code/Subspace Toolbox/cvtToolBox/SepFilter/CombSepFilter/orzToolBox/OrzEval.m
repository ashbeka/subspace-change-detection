
classdef OrzEval
    properties (SetAccess = public)
        ER;
        EER;
        
        A;
        FAR;
        FRR;
        
        nP;
        nN;
        
        flgSIM;
        AUC;
        
    end% properties
    
    methods
        function OB = OrzEval(VAL, Label, varargin)
%function OB = OrzEval(VAL, Label, varargin)
% VAL:      —ÞŽ—“x‚à‚µ‚­‚Í”ñ—ÞŽ—“x?i‹——£?j‚ª“ü‚Á‚½?s—ñ?A‚à‚µ‚­‚Í?sƒxƒNƒgƒ‹
%           VAL‚ª?s—ñ‚Ì?ê?‡?A‘½ƒNƒ‰ƒX–â‘è?i‚QƒNƒ‰ƒXˆÈ?ã?j‚Æ”»’f
%           VAL‚ª?sƒxƒNƒgƒ‹‚Ì?ê?‡?A‚PƒNƒ‰ƒX‚Æ”»’f?ËER‚ðŒvŽZ‚µ‚È‚¢
% Label:    VAL‚Ì—ñ?”‚Æ“¯‚¶ƒTƒCƒY‚Ì?sƒxƒNƒgƒ‹
%           VAL‚Ì?³‰ðƒ‰ƒxƒ‹‚ð•ÛŽ?
%           ‘½ƒNƒ‰ƒX–â‘è‚Ì?ê?‡?A‚P?`ƒNƒ‰ƒX?”‚Ì’l
%           ‚PƒNƒ‰ƒX–â‘è‚Ì?ê?‡?A‚P?iPositive?j‚Æ‚O?iNegative?j
% ‘æŽOˆø?”?F VAL‚Ì’l‚ª—ÞŽ—“x‚©”ñ—ÞŽ—“x?i‹——£?j‚ðŒˆ’è‚·‚é
%           ƒfƒtƒHƒ‹ƒg‚Å‚Í?A—ÞŽ—“x
%           •¶Žš'D'‚ª‘æŽOˆø?”‚É“ü—Í‚³‚ê‚½?ê?‡?A”ñ—ÞŽ—“x?i‹——£?j‚Æ‚µ‚ÄŒvŽZ
%           
% PlotEER?F False Reject Rate‚ÆFalse Alarm  Rate‚ðFigure(10)‚É•`‰æ
%           ˆø?”‚É‚æ‚è?A”Ô?†‚ð•Ï?X‰Â”\
% PlotROC?F ROC curve‚ðFigure(100)‚É•`‰æ
%           ˆø?”‚É‚æ‚è?A”Ô?†‚ð•Ï?X‰Â”\
            
            VAL=VAL(:,:);
            % —ÞŽ—“x‚©”ñ—ÞŽ—“x‚©
            OB.flgSIM=true;
            if nargin == 3
                if varargin{1}=='D';
                    OB.flgSIM=false;
                end
            end
            
            % One-Class –â‘è‚©‚Ç‚¤‚©
            if size(VAL,1)>=2
                B=zeros(size(VAL));
                Lu = unique(Label);
                for I=1:size(Lu,2)
                    B(I,Label==Lu(I))=1;
                end
                
                if OB.flgSIM
                    [v ind] = max(VAL,[],1);
                else
                    [v ind] = min(VAL,[],1);
                end
                OB.ER = 1-mean(ind == Label);
                
            else
                B = zeros(size(Label));
                B(Label~=0)=1;
            end
            VAL=VAL(:);
            B=B(:);
            
            OB.nP = sum(B==1);
            OB.nN = sum(B==0);
            
            if OB.flgSIM
                [OB.A C]= sort(VAL,'ascend');
            else
                [OB.A C]= sort(VAL,'descend');
            end
            D = B(C);
            
            OB.FAR = 1-cumsum(D==0)/OB.nN;
            OB.FRR = cumsum(D==1)/OB.nP;
            
            [val ind] = min((abs(OB.FAR-OB.FRR)));
            OB.EER = (OB.FAR(ind)+OB.FRR(ind))/2;
            
        end
        
        function PlotEER(OB,varargin)
            if nargin == 2
                No = varargin{1};
            else
                No = 10;
            end
            
            figure(No)
            clf;
            hold on
            plot(OB.A,OB.FRR,'b');
            plot(OB.A,OB.FAR,'r');
            title('FRR - FAR');
            legend('False Reject Rate','False Alarm  Rate');
            xlabel('Threshold')
            ylabel('Rate')
            hold off
        end
        
        function OB = PlotROC(OB,varargin)
            if nargin == 2
                No = varargin{1};
                color = 'r';
            elseif nargin == 3
                No = varargin{1};
                color = varargin{2};   
            else
                No = 100;
                color = 'r';
            end
            
            figure(No)
            %clf;
            hold on
            plot(OB.FAR,1-OB.FRR,color);
            xlabel('False Positive Rate')
            ylabel('True Positive Rate')
            hold off
            OB.AUC = sum(mean(OB.FAR(1:end-1)-OB.FAR(2:end)) .* (1-OB.FRR));
        end
    end
end