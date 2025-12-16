classdef CvtKCMSM
    properties (SetAccess = private)
        nDim;
        nNum;
        nClass;
        nOrthSubDim;
        nOrthDim;
        Sigma2;
        
        nAlpha;
        
        
        X;
        KO;
        A;
        EV;
        BETA;
        C_RATE;
        
        U;
        
    end% properties
    
    methods
        function obj = train(obj, X, nOrthSubDim,Sigma2)
            
            obj.X = X;
            obj.nOrthSubDim = nOrthSubDim;
            obj.Sigma2 = Sigma2;
            
            [obj.nDim , obj.nNum, obj.nClass] = size(X);
            [obj.A, obj.EV, SSS, obj.C_RATE] =  cvtKernelBasisVector(obj.X, obj.nOrthSubDim, obj.Sigma2);
            
            D = zeros( obj.nOrthSubDim, obj.nOrthSubDim, obj.nClass, obj.nClass);
            for I1 = 1: obj.nClass
                for I2 = I1: obj.nClass
                    if I1 == I2
                        D(:,:,I1,I2) = eye( obj.nOrthSubDim, obj.nOrthSubDim);
                    else
                        K12 = cvtKernelMatrix(obj.X(:,:,I1), obj.X(:,:,I2), obj.Sigma2);
                        tmpD =  obj.A(:,:,I1)' * K12 * obj.A(:,:,I2);
                        D(:,:,I1,I2) = tmpD;
                        D(:,:,I2,I1) = tmpD';
                    end
                end
            end
            D = reshape(permute(D,[1,3,2,4]), obj.nOrthSubDim * obj.nClass, obj.nOrthSubDim * obj.nClass);
            

            obj.nOrthDim = obj.nOrthSubDim * obj.nClass;
            
            [B,C] = cvtEig(D);
            C = C / obj.nClass;
            B = B * diag(1./sqrt(C));
            obj.KO = B;
            obj.BETA = C';
                       
        end
        
        function [OY] = predict(obj, Y, nInSubDim)
            Y=Y(:,:,:);
            OY = zeros(obj.nOrthDim,nInSubDim,size(Y,3));

            [A, EV, SSS, C_RATE] = cvtKernelBasisVector(Y, nInSubDim, obj.Sigma2);
            
            for J = 1:size(A,3)
                a = zeros( obj.nOrthSubDim, size(Y,2),obj.nClass);
                for I = 1:obj.nClass
                    Z = cvtKernelMatrix( obj.X(:,:,I), Y(:,:,J), obj.Sigma2);
                    a(:,:,I) = obj.A(:,:,I)' * Z;
                end
                a = permute(a,[1,3,2]);
                a = reshape(a,size(a,1)*size(a,2),size(a,3));
                OY(:,:,J) = (obj.KO'*a*A(:,:,J));
                
                
                
            end
        end
    end% methods
end% classdef
% classdef CvtKCMSM
%     properties (SetAccess = private)
%         nDim;
%         nNum;
%         nClass;
%         nOrthSubDim;
%         nOrthDim;
%         Sigma2;
%         
%         nAlpha;
%         
%         
%         X;
%         KO;
%         A;
%         EV;
%         BETA;
%         C_RATE;
%         
%         U;
%         
%     end% properties
%     
%     methods
%         function obj = CvtKCMSM(X, nOrthSubDim,Sigma2)
%             
%             obj.X = X;
%             obj.nOrthSubDim = nOrthSubDim;
%             obj.Sigma2 = Sigma2;
%             
%             [obj.nDim , obj.nNum, obj.nClass] = size(X);
%             [SSS, obj.A, obj.C_RATE, obj.EV] =  cvtKernelBasisVector(obj.X, obj.nOrthSubDim, obj.Sigma2);
%             
%             D = zeros( obj.nOrthSubDim, obj.nOrthSubDim, obj.nClass, obj.nClass);
%             for I1 = 1: obj.nClass
%                 for I2 = I1: obj.nClass
%                     if I1 == I2
%                         D(:,:,I1,I2) = eye( obj.nOrthSubDim, obj.nOrthSubDim);
%                     else
%                         K12 = cvtKernelMatrix( obj.X(:,:,I1), obj.X(:,:,I2), obj.Sigma2);
%                         tmpD =  obj.A(:,:,I1)'*K12* obj.A(:,:,I2);
%                         D(:,:,I1,I2) = tmpD;
%                         D(:,:,I2,I1) = tmpD';
%                     end
%                 end
%             end
%             D = reshape(permute(D,[1,3,2,4]), obj.nOrthSubDim * obj.nClass, obj.nOrthSubDim * obj.nClass);
%             
% 
%             obj.nOrthDim = obj.nOrthSubDim * obj.nClass;
%             
%             [B,C] = cvtEig(D);
%             C = C / obj.nClass;
%             B = B * diag(1./sqrt(C));
%             obj.KO = B;
%             obj.BETA = C';
%                        
%         end
%         
%         function [OY ]= Transform(obj, Y, nInSubDim)
%             Y=Y(:,:,:);
%             OY = zeros(obj.nOrthDim,nInSubDim,size(Y,3));
%             [SSS, A, C_RATE, EV] =  cvtKernelBasisVector(Y, nInSubDim, obj.Sigma2);
%             for J = 1:size(A,3)
%                 a = zeros( obj.nOrthSubDim, size(Y,2),obj.nClass);
%                 for I = 1:obj.nClass
%                     Z = cvtKernelMatrix( obj.X(:,:,I), Y(:,:,J), obj.Sigma2);
%                     a(:,:,I) = obj.A(:,:,I)' * Z;
%                 end
%                 a = permute(a,[1,3,2]);
%                 a = reshape(a,size(a,1)*size(a,2),size(a,3));
%                 OY(:,:,J) = (obj.KO'*a*A(:,:,J));
%             end
%         end
%     end% methods
% end% classdef
