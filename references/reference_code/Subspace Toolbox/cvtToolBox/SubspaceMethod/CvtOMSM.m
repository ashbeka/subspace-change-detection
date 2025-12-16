classdef CvtOMSM
    properties
        train_subspace
        train_orth_subspace
        orth_subspace
        
        train_orth_subspaces
        orth_subspaces

    end
    
    methods
        function obj = train(obj, train_x, subspace_dim, flag_all)
            % train_data:学習データ
            % train_label:学習データのラベル
            % subspace_dim:辞書部分空間の次元数   
            
            if nargin < 5
                flag_all = true;
            end
            
            train_x = cvtNormalize(train_x); % ノルムを１にする
            obj.train_subspace = cvtBasisVectorSVD(train_x, subspace_dim);

            [nDim, ~, nClass] = size(train_x);

            
            % Generalizated orthogonal subspace
            P = obj.train_subspace(:,:) * obj.train_subspace(:,:)';             
            [B C] = eig(P);
            C = diag(C);
            [~, index] = sort(C,'descend');
            B = B(:,index);C = C(index);
            obj.orth_subspace = (sqrt(inv(diag(C(1:rank(P)))))*B(:,1:rank(P))')';
            
           
            obj.train_orth_subspace = zeros(size(obj.orth_subspace,2), subspace_dim, nClass);
            for iClass = 1:nClass
                obj.train_orth_subspace(:,:,iClass) = orth(obj.orth_subspace'*obj.train_subspace(:,:,iClass));
            end
            
            
            if flag_all == true
                obj.orth_subspaces = cell(subspace_dim);
                obj.train_orth_subspaces = cell(subspace_dim);
                for iDim = 1:subspace_dim
                    tmp_train_subspace = obj.train_subspace(:,1:iDim,:);
                    P =  tmp_train_subspace(:,:) * tmp_train_subspace(:,:)';
                    [B C] = eig(P);
                    C = diag(C);
                    [~, index] = sort(C,'descend');
                    B = B(:,index);C = C(index);
                    tmp_orth_subspace = (sqrt(inv(diag(C(1:rank(P)))))*B(:,1:rank(P))')';
                    
                    nDim_orth_subspace = size(tmp_orth_subspace, 2);
                    tmp_train_orth_subspace = zeros(nDim_orth_subspace, iDim, nClass);
                    for iClass = 1:nClass
                        tmp_train_orth_subspace(:,:,iClass) = orth(tmp_orth_subspace'*tmp_train_subspace(:,:,iClass));
                    end
                    
                    obj.orth_subspaces{iDim} = tmp_orth_subspace;
                    obj.train_orth_subspaces{iDim} = tmp_train_orth_subspace;
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
            input_orth_subspace = zeros(size(obj.orth_subspace,2), subspace_dim, data_num);
            for di = 1:data_num
                input_orth_subspace(:,:,di) = orth(obj.orth_subspace'*input_subspace(:,:,di));
            end
            
            sim = cvtCanonicalAnglesMean(obj.train_orth_subspace, input_orth_subspace);
            
            [~, predict_index] = max(sim, [], 1);
    
        end
        
        function [sim predict_index] = predictMany(obj, input_x, max_subspace_dim)
%             
%             input_x = input_x(:,:,:);
%             [data_dim, ~,data_num] = size(input_x);
%             
%             input_x = cvtNormalize(input_x);
%             input_subspace = cvtBasisVectorSVD(input_x, max_subspace_dim);
%             input_orth_subspace = zeros(size(obj.orth_subspace,2), max_subspace_dim, data_num);
%             for di = 1:data_num
%                 input_orth_subspace(:,:,di) = orth(obj.orth_subspace'*input_subspace(:,:,di));
%             end
%             
%             sim = cvtCanonicalAnglesMeanMany(obj.train_orth_subspace, input_orth_subspace);
%             
%             nSubInput = size(input_x, 3);
%             nDimDic = size(obj.train_orth_subspace, 2);
%             
%             predict_index = zeros(nSubInput, nDimDic, max_subspace_dim);
%             
%             for iDimSubX = 1:size(obj.train_orth_subspace, 2)
%             for iDimSubY = 1:max_subspace_dim
%                 [~, predict_index(:,iDimSubX, iDimSubY)] = max(sim(:,:,iDimSubX,iDimSubY), [], 1);
%             end
%             end
%             
            input_x = input_x(:,:,:);
            [data_dim, ~,data_num] = size(input_x);
            
            input_x = cvtNormalize(input_x);
            input_subspace = cvtBasisVectorSVD(input_x, max_subspace_dim);
          
            [~,nDimDic,nDic] = size(obj.train_orth_subspace);
            sim =  zeros(nDic, data_num, nDimDic, max_subspace_dim);
            predict_index = zeros(data_num, nDimDic, max_subspace_dim);
            
            for iDimInput = 1:max_subspace_dim
                tmp_input_subspace = input_subspace(:,1:iDimInput,:);                
                for iDimDic = 1:nDimDic
                    tmp_orth_subspace = obj.orth_subspaces{iDimDic};
                    tmp_train_orth_subspace = obj.train_orth_subspaces{iDimDic};
                    input_orth_subspace = zeros(size(tmp_orth_subspace,2), iDimInput, data_num);

                    for di = 1:data_num
                        input_orth_subspace(:,:,di) ...
                            = orth(tmp_orth_subspace'*tmp_input_subspace(:,:,di));
                    end

                    tmp_sim = cvtCanonicalAnglesMean(tmp_train_orth_subspace, input_orth_subspace);
                    sim(:,:,iDimDic,iDimInput) = tmp_sim;
                    [~, predict_index(:,iDimDic, iDimInput)] = max(tmp_sim, [], 1);
                end
            end
        end        
    end
end
