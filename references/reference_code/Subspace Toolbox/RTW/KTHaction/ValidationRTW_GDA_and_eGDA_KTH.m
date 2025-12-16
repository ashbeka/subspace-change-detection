clear
load('Z:\Lincon\Datasets\KTH_action_dataset\KTH_dataset.mat')
DATA = permute(kth_data,[1 2 4 3]); % becomes [persons conditions repetitions classes]
X1 = DATA(:,:,1:2,:);
X2 = DATA(:,:,3:4,:);

[X1,TrnLabels] = processKTHdata_RTWexperiments(X1);
[X2,TestLabels] = processKTHdata_RTWexperiments(X2);

nSelectedFrames = 15;
n_TEfeatures = 100;
% SubDim = 10;
% ClassSubDim = 30;
% nDimPS = 10;
% Accuracy = deep_RTWmachine(X1,X2,r,n,Net);
%------------pre-processing
% DATA2 = cellfun(@(x) orzReshape(x,1),DATA, 'UniformOutput', false);
% X1 = DATA2(1:10,:,:);
% X2 = DATA2(11:20,:,:);
%
% X1 = orzReshape(X1,1);
% X2 = orzReshape(X2,1);


% GDAtable = {'Accuracy' 'SubDim' 'TrnTime' 'TestTime'};
% eGDAtable = {'Accuracy' 'SubDim' 'ClassSubDim' 'DimPS' 'TrnTime' 'TestTime'};

GDAtable = {'Accuracy' 'SubDim'};
eGDAtable = {'Accuracy' 'SubDim' 'ClassSubDim' 'DimPS'};

ListSubDim = [7:15];
ListClassSubDim = [90:10:120];
ListDimPS = [5:5:20];



% load Permutations4crossval % fold 10, remainingPerms, computerFolds
% load ValRawLogSetNet_ALLFOLDS % results obtained from validation
% Summary = GenSummary(ValRawLog);
% tmp = Summary.MethodsDetails{2,1}.foldBestResults;

% for fold = 1:10 % ALL FOLDS (not divided by machine)
%     TrnFoldNum = fold10(fold,1);
%     ValFoldNum = fold10(fold,2);
%     TestFoldNum = fold10(fold,3:4);

% prepare the data for this fold
% [TrnData,ValData,TrnLabels,ValLabels] = CMUexpressionsDataOrganize(...
%     sessions,TrnFoldNum,ValFoldNum);
% nTrnImg = length(TrnLabels.class);
% nValImg = length(ValLabels.class);
% X = listArray2dividedArray(orzReshape(TrnData,1),TrnLabels,nTrnImg);
% Y = listArray2dividedArray(orzReshape(ValData,1),ValLabels,nValImg);

%     for PatchSize = ListPatchSize
% %         for NumFilters = ListNumFilters


for SubDim = ListSubDim
    %
    %     %% Conventional GDA
    %                     tic
    skipFlag = Check4Redundancy(GDAtable,...
        {SubDim},[2]);
    if skipFlag == 1 % if this combination is already done
        continue
    end

    %     [Accuracy,Time] = GrasMachineNewKNN(X,Y,SubDim);
    idx = find(SubDim == ListSubDim);
    %     [Accuracy,V1{idx},V2{idx}] = RTWmachine(X1,X2,nSelectedFrames,n_TEfeatures,SubDim);
    [Accuracy,V1{idx},V2{idx}] = RTWmachine(X1,X2,TrnLabels,TestLabels,nSelectedFrames,n_TEfeatures,SubDim);

    %     eGDAtable = append2Table(eGDAtable,{Accuracy,SubDim});
    GDAtable = append2Table(GDAtable,{Accuracy,SubDim});

    %                 toc
    %             end
    %         end
end

% for SubDim = ListSubDim
%     for ClassSubDim = ListClassSubDim
%         for DimPS = ListDimPS
%             
%             
%             %% eGDA
%             %                     tic
%             skipFlag = Check4Redundancy(eGDAtable,...
%                 {SubDim ClassSubDim DimPS},[2:4]);
%             if skipFlag == 1 % if this combination is already done
%                 continue
%             end
%             
%             %     [Accuracy,Time] = GrasMachineNewKNN(X,Y,SubDim);
%             idx = find(SubDim == ListSubDim);
%             %             [Accuracy,V1{idx},V2{idx}] = RTWmachine(X1,X2,nSelectedFrames,n_TEfeatures,SubDim);
%             %             Accuracy = GDS_RTWmachine(X1,X2,nSelectedFrames,n_TEfeatures,SubDim,ClassSubDim,DimPS);
%             %             Accuracy = simplifiedGDS_RTWmachine(V1{idx},V2{idx},ClassSubDim,DimPS);
%             Accuracy = GDS_RTWmachine(X1,X2,TrnLabels,TestLabels,nSelectedFrames,n_TEfeatures,SubDim,ClassSubDim,DimPS);
%             %             Accuracy = simplifiedGDS_RTWmachine(V1{idx},V2{idx},TrnLabels,TestLabels,ClassSubDim,DimPS);
%             
%             %     eGDAtable = append2Table(eGDAtable,{Accuracy,SubDim});
%             eGDAtable = append2Table(eGDAtable,{Accuracy,SubDim,ClassSubDim,DimPS});
%             
%             %                 toc
%         end
%     end
% end


% JUST ONE METHOD
ValRawLog{1,1} = 'RTW+GDA';
ValRawLog{2,1} = GDAtable;

% zz = 1;
% ValRawLog{1,zz} = 'RTW+eGDA';
% ValRawLog{2,zz} = eGDAtable;


Summary = GenSummary(ValRawLog);


