
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
        
    end% properties
    
    methods
        function OB = OrzEval(VAL, Label, varargin)
%function OB = OrzEval(VAL, Label, varargin)
% VAL:      類似度もしくは非類似度（距離）が入った行列、もしくは行ベクトル
%           VALが行列の場合、多クラス問題（２クラス以上）と判断
%           VALが行ベクトルの場合、１クラスと判断⇒ERを計算しない
% Label:    VALの列数と同じサイズの行ベクトル
%           VALの正解ラベルを保持
%           多クラス問題の場合、１～クラス数の値
%           １クラス問題の場合、１（Positive）と０（Negative）
% 第三引数： VALの値が類似度か非類似度（距離）を決定する
%           デフォルトでは、類似度
%           文字'D'が第三引数に入力された場合、非類似度（距離）として計算
%           
% PlotEER： False Reject RateとFalse Alarm  RateをFigure(10)に描画
%           引数により、番号を変更可能
% PlotROC： ROC curveをFigure(100)に描画
%           引数により、番号を変更可能
            
            VAL=VAL(:,:);
            % 類似度か非類似度か
            OB.flgSIM=true;
            if nargin == 3
                if varargin{1}=='D';
                    OB.flgSIM=false;
                end
            end
            
            % One-Class 問題かどうか
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
            legend('False Reject Rate','False Alarm  Rate',0);
            xlabel('Threshold')
            ylabel('Rate')
            hold off
        end
        
        function PlotROC(OB,varargin)
            if nargin == 2
                No = varargin{1};
            else
                No = 100;
            end
            
            figure(No)
            clf;
            hold on
            plot(OB.FAR,1-OB.FRR,'k');
            xlabel('False Positive Rate')
            ylabel('True Positive Rate')
            hold off
        end
    end
end