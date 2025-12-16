classdef OrzKOMSM
    properties (SetAccess = public)
        nDim;
        nNum1;
        nClass;
        nSubDim1;
        nOrthDim;
        nSigma;
                
        nAlpha;
        nBeta;
        
        X1;
        A1;
        E1;
        C1;
        
        D;
        O;
        W;
    end% properties
    
    methods
        function OB = OrzKOMSM(X1, nSubDim1,nSigma, varargin)
%function OB = OrzKOMSM(X1, nSubDim1,nSigma, varargin)
% nDim:     次元
% nNum1:    各クラスのデータ数
% nClass:   クラス数
% nSubDim1: 辞書部分空間の時限
% nOrthDim: 直交化空間の時限
% nSigma:   ガウシアンカーネルパラメータ
% nAlpha:   直交化行列の寄与率(第４パラメータ、デフォルトで1.0)
%             
% X1:       辞書データ
% A1:       X1のKPCA結合係数
% E1:       固有値
% C1:       寄与率
%             
% D:        直交化カーネルグラム行列
% O:        カーネル直交化行列
% W:        Dの固有値
%
% [V2 E2 C2 A2 ] = TransformS(OB, X2,nSubDim2)
% 部分空間に対する直交化変換
% KPCA ⇒ 直交化変換 ⇒ グラムシュミット
%
% [V2 E2 C2 Y2 ] = TransformV(OB, X2, nSubDim2)
% パターンベクトルに対する直交化変換し基底ベクトルを求める
% 非線形直交化変換 ⇒ PCA
%
%function [Y2] = Transform(OB, X2)
% パターンベクトルに対する直交化変換だけ
% 非線形直交化変換
     
      
            if nargin == 3
                OB.nAlpha = 1;
            else
                OB.nAlpha = varargin{1};
            end
            
            OB.nSubDim1 = nSubDim1;
            OB.nSigma = nSigma;
            OB.X1 = X1;
            
            
            [OB.nDim , OB.nNum1, OB.nClass] = size(X1);
            
            OB.A1 = zeros(OB.nNum1,OB.nSubDim1,OB.nClass);
            OB.E1 = zeros(OB.nSubDim1,OB.nClass);
            OB.C1 = zeros(1,OB.nClass);
            for I=1:OB.nClass
                [OB.A1(:,:,I) OB.E1(:,I) OB.C1(I)] = orzKPCA(OB.X1(:,:,I),OB.nSubDim1,OB.nSigma,'R');
                I
            end
          
            OB.D = zeros(OB.nSubDim1, OB.nSubDim1, OB.nClass,OB.nClass);
            for I1 = 1:OB.nClass
                for I2 = I1:OB.nClass
                    K = exp(-orzL2Distance(X1(:,:,I1),X1(:,:,I2))/nSigma);
                    if I1 == I2
                        OB.D(:,:,I1,I2) = eye(OB.nSubDim1,OB.nSubDim1);
                    else
                        OB.D(:,:,I1,I2) =  OB.A1(:,:,I1)'*K* OB.A1(:,:,I2);
                        OB.D(:,:,I2,I1) = OB.D(:,:,I1,I2)';
                    end
                end
            end
            OB.D = reshape(permute(OB.D,[1,3,2,4]), OB.nSubDim1 * OB.nClass, OB.nSubDim1 * OB.nClass);
            
            [B,OB.W] = svd(OB.D);%[B,OB.W] = eig(OB.D);            
            OB.W=diag(OB.W/trace(OB.W));
            
            OB.nOrthDim = find(cumsum(OB.W)/sum(OB.W)>=OB.nAlpha, 1 );
            B=B(:,1:OB.nOrthDim);
            OB.W = OB.W(1:OB.nOrthDim);
                    
            OB.O = diag(1./(OB.W))*B';

        end
        function [V2 E2 C2 A2 ] = TransformS(OB, X2,nSubDim2)
            % 部分空間に対する直交化変換
            % KPCA ⇒ 直交化変換 ⇒ グラムシュミット
            nSize = size(X2);
            X2=X2(:,:,:);
            [tmp, nNum2,nSet2] = size(X2);
            
            A2 = zeros(nNum2,nSubDim2,nSet2);
            E2 = zeros(nSubDim2,nSet2);
            C2 = zeros(1,nSet2);
            for I=1:nSet2
                [A2(:,:,I) E2(:,I) C2(I)] = orzKPCA(X2(:,:,I),nSubDim2,OB.nSigma,'R');
            end
            
            V2 = zeros(OB.nOrthDim,nSubDim2,nSet2);
            for J = 1:nSet2
                a = zeros( OB.nSubDim1, nNum2,OB.nClass);
                for I = 1:OB.nClass
                    Z = exp(-orzL2Distance(OB.X1(:,:,I),X2(:,:,J))/OB.nSigma);
                    
                    a(:,:,I) = OB.A1(:,:,I)' * Z;
                end
                a = permute(a,[1,3,2]);
                a = reshape(a,size(a,1)*size(a,2),size(a,3));
                V2(:,:,J) = cvtGramSchmidt(OB.O*a*A2(:,:,J));
            end
            V2 = reshape(V2,[OB.nOrthDim,nSubDim2,nSize(3:end),1]);
        end
        
        
        function [V2 E2 C2 Y2 ] = TransformV(OB, X2, nSubDim2)
            % パターンベクトルに対する直交化変換し基底ベクトルを求める
            % 非線形直交化変換 ⇒ PCA
            nSize = size(X2);
            X2=X2(:,:,:);
            [tmp nNum2,nSet2] = size(X2);
            
            Y2 = zeros(OB.nOrthDim,nNum2,nSet2);
            for J=1:nSet2
                a = zeros( OB.nSubDim1,nNum2,OB.nClass);
                for I = 1:OB.nClass
                    Z = exp(-orzL2Distance(OB.X1(:,:,I),X2(:,:,J))/OB.nSigma);
                    a(:,:,I) = OB.A1(:,:,I)' * Z;
                end
                a = permute(a,[1,3,2]);
                a = reshape(a,OB.nSubDim1*OB.nClass,nNum2);
                Y2(:,:,J) = OB.O*a;
            end
            
            V2 = zeros(OB.nOrthDim,nSubDim2,nSet2);
            E2 = zeros(nSubDim2,nSet2);
            C2 = zeros(nSet2);
            for J=1:nSet2
                [tmp V2(:,:,J) E2(:,J) C2(J)] = orzPCA(Y2(:,:,J),nSubDim2,'R');
            end
            
            Y2 = reshape(Y2,[OB.nOrthDim,nNum2,nSize(3:end),1]);
            V2 = reshape(V2,[OB.nOrthDim,nSubDim2,nSize(3:end),1]);
            
        end
                
        function [V2 E2 C2 Y2 ] = TransformU(OB, X2, nSubDim2)
            % パターンベクトルに対する直交化変換し基底ベクトルを求める
            % 非線形直交化変換 ⇒ ノルムの正規化 ⇒ PCA
            nSize = size(X2);
            X2=X2(:,:,:);
            [tmp nNum2,nSet2] = size(X2);
            
            Y2 = zeros(OB.nOrthDim,nNum2,nSet2);
            for J=1:nSet2
                a = zeros( OB.nSubDim1,nNum2,OB.nClass);
                for I = 1:OB.nClass
                    Z = exp(-orzL2Distance(OB.X1(:,:,I),X2(:,:,J))/OB.nSigma);
                    a(:,:,I) = OB.A1(:,:,I)' * Z;
                end
                a = permute(a,[1,3,2]);
                a = reshape(a,OB.nSubDim1*OB.nClass,nNum2);
                Y2(:,:,J) = OB.O*a;
            end
            
            Y2 = orzNormalize(Y2);
            V2 = zeros(OB.nOrthDim,nSubDim2,nSet2);
            E2 = zeros(nSubDim2,nSet2);
            C2 = zeros(nSet2);
            for J=1:nSet2
                [tmp V2(:,:,J) E2(:,J) C2(J)] = orzPCA(Y2(:,:,J),nSubDim2,'R');
            end
            
            Y2 = reshape(Y2,[OB.nOrthDim,nNum2,nSize(3:end),1]);
            V2 = reshape(V2,[OB.nOrthDim,nSubDim2,nSize(3:end),1]);
            
        end
  
        function [Y2] = Transform(OB, X2)
            % パターンベクトルに対する直交化変換だけ
            % 非線形直交化変換
            nSize = size(X2);
            X2=X2(:,:,:);
            [tmp nNum2,nSet2] = size(X2);
            
            Y2 = zeros(OB.nOrthDim,nNum2,nSet2);
            for J=1:nSet2
                a = zeros( OB.nSubDim1,nNum2,OB.nClass);
                for I = 1:OB.nClass
                    Z = exp(-orzL2Distance(OB.X1(:,:,I),X2(:,:,J))/OB.nSigma);
                    
                    a(:,:,I) = OB.A1(:,:,I)' * Z;
                end
                a = permute(a,[1,3,2]);
                a = reshape(a,OB.nSubDim1*OB.nClass,nNum2);
                Y2(:,:,J) = OB.O*a;
            end
            Y2 = reshape(Y2,[OB.nOrthDim,nNum2,nSize(3:end),1]);
            
        end
        
    end% methods
end% classdef
