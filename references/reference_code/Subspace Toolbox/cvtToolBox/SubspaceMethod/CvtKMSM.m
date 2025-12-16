classdef CvtKMSM
    properties
        DicV
        DicData
    end
    
    methods
        function obj = train(obj, train_data, subspace_dim, sigma2)
            % train_data:学習データ
            % subspace_dim:辞書部分空間の次元
            
            [dim, each_data_num, class_num] = size(train_data);

            train_data = cvtNormalize(train_data); % ノルムを１にする
      
            obj.DicV = zeros(each_data_num, subspace_dim, class_num);      
            for I=1:class_num
                [obj.DicV(:,:,I), ~] = cvtKernelBasisVector(train_data(:,:,I), subspace_dim, sigma2);
            end
            
            obj.DicData = train_data;
        end
        
        function [sim] = predict(obj, data, subspace_dim, sigma2)
            % data:入力データ size:データの次元x各データの個数xデータの個数
            % subpsace_dim:入力の部分空間の次元
            
            [data_dim, each_data_num, data_num] = size(data);
            data = cvtNormalize(data);
            
            inpV = zeros(each_data_num, subspace_dim, data_num);
            for di = 1:data_num
                inpV(:,:,di) = cvtKernelBasisVector(data(:,:,di), subspace_dim, sigma2);
            end
            
            num_class = size(obj.DicV, 3);
            sim = zeros(num_class, data_num);
            
            for di = 1:data_num
                for ci = 1:num_class
                    % cos( Canonical Angles )
                    K = cvtKernelMatrix(obj.DicData(:,:,ci), data(:,:,di), sigma2);
                    cos_angle = svd(obj.DicV(:,:,ci)'*K*inpV(:,:,di));
                    
                    % similarity = mean(cos^2(c_angles))
                    sim(ci,di) = mean(cos_angle.^2);
                end
            end
        end
    end
    
    
end
