classdef CvtCMSM
    properties
        train_subspace
        train_diff_subspace
        diff_subspace

        train_diff_subspaces
        diff_subspaces
    
    end
    
    methods
        function obj = train(obj, train_x, subspace_dim, del_subspace_dim, flag_all)
            % train_data:学習データ
            % subspace_dim:辞書部分空間の次元数
            % del_subspace_dim:差分部分空間を作るときに消す次元数
           
            if nargin < 5
                flag_all = true;
            end
            
            train_x = cvtNormalize(train_x); % ノルムを１にする
            obj.train_subspace = cvtBasisVectorSVD(train_x, subspace_dim);
            
            [nDim, ~, nClass] = size(train_x);
            
            % Generalizated difference subspace(Constraint Subspace)
            P = obj.train_subspace(:,:) * obj.train_subspace(:,:)';
            
%             [B C] = eig(P);
%             C = diag(C);
%             [~, index] = sort(C,'descend');
%             B = B(:,index); C = C(index);
%             obj.diff_subspace = B(:,del_subspace_dim+1:rank(P));
            tmp_diff_subspace = cvtBasisVectorSVD(P, rank(P));
            obj.diff_subspace = tmp_diff_subspace(:, (del_subspace_dim+1):end);
            
           
            nDim_diff_subspace = size(obj.diff_subspace, 2);
            obj.train_diff_subspace = zeros(nDim_diff_subspace, subspace_dim, nClass);
            for iClass = 1:nClass
                obj.train_diff_subspace(:,:,iClass) = orth(obj.diff_subspace'*obj.train_subspace(:,:,iClass));
            end
            
            
            if flag_all == true
                obj.diff_subspaces = cell(subspace_dim);
                obj.train_diff_subspaces = cell(subspace_dim);
                for iDim = 1:subspace_dim
                    tmp_train_subspace = obj.train_subspace(:,1:iDim,:);
                    P =  tmp_train_subspace(:,:) * tmp_train_subspace(:,:)';
                    
                    tmp_diff_subspace = cvtBasisVectorSVD(P, rank(P));
                    tmp_diff_subspace = tmp_diff_subspace(:, (del_subspace_dim+1):end);                    

                    nDim_diff_subspace = size(tmp_diff_subspace, 2);
                    tmp_train_diff_subspace = zeros(nDim_diff_subspace, iDim, nClass);
                    for iClass = 1:nClass
                        tmp_train_diff_subspace(:,:,iClass) = orth(tmp_diff_subspace'*tmp_train_subspace(:,:,iClass));
                    end
                    
                    obj.diff_subspaces{iDim} = tmp_diff_subspace;
                    obj.train_diff_subspaces{iDim} = tmp_train_diff_subspace;
                end
            end            
        end
        
        function [sim predict_index] = predict(obj, input_x, subspace_dim)
            % data:入力データ size:データの次元x各データの個数xデータの個数
            % subpsace_dim:入力の部分空間の次元
            
            input_x = input_x(:,:,:);
            [data_dim, ~,data_num] = size(input_x);
            input_x = cvtNormalize(input_x);
    
            input_subspace = cvtBasisVectorSVD(input_x, subspace_dim);
            input_diff_subspace = zeros(size(obj.diff_subspace,2), subspace_dim, data_num);
            
            for di = 1:data_num
                input_diff_subspace(:,:,di) = orth(obj.diff_subspace'*input_subspace(:,:,di));
            end
            
            sim = cvtCanonicalAnglesMean(obj.train_diff_subspace, input_diff_subspace);            
           
            [~, predict_index] = max(sim, [], 1);
        end
        
        function [sim predict_index] = predictMany(obj, input_x, max_subspace_dim)
            
            input_x = input_x(:,:,:);
            [data_dim, ~,data_num] = size(input_x);
            
            input_x = cvtNormalize(input_x);
            input_subspace = cvtBasisVectorSVD(input_x, max_subspace_dim);
          
            [~,nDimDic,nDic] = size(obj.train_diff_subspace);

            sim =  zeros(nDic, data_num, nDimDic, max_subspace_dim);
            predict_index = zeros(data_num, nDimDic, max_subspace_dim);
            for iDimInput = 1:max_subspace_dim
                tmp_input_subspace = input_subspace(:,1:iDimInput,:);                
                for iDimDic = 1:nDimDic
                    tmp_diff_subspace = obj.diff_subspaces{iDimDic};
                    tmp_train_diff_subspace = obj.train_diff_subspaces{iDimDic};
                    input_diff_subspace = zeros(size(tmp_diff_subspace,2), iDimInput, data_num);

                    for di = 1:data_num
                        input_diff_subspace(:,:,di) ...
                            = orth(tmp_diff_subspace'*tmp_input_subspace(:,:,di));
                    end

                    tmp_sim = cvtCanonicalAnglesMean(tmp_train_diff_subspace, input_diff_subspace);
                    sim(:,:,iDimDic,iDimInput) = tmp_sim;
                    [~, predict_index(:,iDimDic, iDimInput)] = max(tmp_sim, [], 1);
                end
            end
        end

        
    end
    
    
end
