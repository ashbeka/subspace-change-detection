 classdef CvtMSM
    properties
        train_subspace
    end
    
    methods
        function obj = train(obj, train_x, subspace_dim)
            % train_data:Šw?Kƒf?[ƒ^
            % train_label:Šw?Kƒf?[ƒ^‚Ìƒ‰ƒxƒ‹
            % subspace_dim:Ž«?‘•”•ª‹óŠÔ‚ÌŽŸŒ³
            
            train_x = cvtNormalize(train_x); % ƒmƒ‹ƒ€‚ð‚P‚É‚·‚é
            obj.train_subspace = cvtBasisVectorSVD(train_x, subspace_dim);
        end
        
        function [sim predict_index] = predict(obj, input_x, subspace_dim)
            % data:“ü—Íƒf?[ƒ^ size:ƒf?[ƒ^‚ÌŽŸŒ³xŠeƒf?[ƒ^‚ÌŒÂ?”xƒf?[ƒ^‚ÌŒÂ?”
            % subpsace_dim:“ü—Í‚Ì•”•ª‹óŠÔ‚ÌŽŸŒ³
            
            input_x = cvtNormalize(input_x);
            input_subspace = cvtBasisVectorSVD(input_x, subspace_dim);
%             disp(size(input_subspace))
%             disp(input_subspace(:,:,1)'*input_subspace(:,:,1));
%             obj.hello=erro
            clear input_x;
            sim = cvtCanonicalAnglesMean(obj.train_subspace, input_subspace);

            [~, predict_index] = max(sim, [], 1);
        end
        
        function [sim predict_index] = predictMany(obj, input_x, max_subspace_dim)
            
            input_x = input_x(:,:,:);
            
            input_x = cvtNormalize(input_x);
            input_subspace = cvtBasisVectorSVD(input_x, max_subspace_dim);
            
            sim = cvtCanonicalAnglesMeanMany(obj.train_subspace, input_subspace);
            
            nSubInput = size(input_x, 3);
            nDimDic = size(obj.train_subspace, 2);
            
            predict_index = zeros(nSubInput, nDimDic, max_subspace_dim);
            
            for iDimSubX = 1:size(obj.train_subspace, 2)
            for iDimSubY = 1:max_subspace_dim
                [~, predict_index(:,iDimSubX, iDimSubY)] = max(sim(:,:,iDimSubX,iDimSubY), [], 1);
            end
            end
        end
    end
    
    
end
