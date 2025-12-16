function [max_model max_data_x] = cvtDBPP_learn(imgs, data_l, s, nPattern)


nIter = 30;
% nPattern = 4;
dim = 2^nPattern;
nImg = size(imgs,3);
score = 0;

max_criterion = 0;

for iIter = 1:nIter

    % ランダムで配置決定
    flag = true;
    while flag == true
        config1 = randi([-s s], 2, nPattern);
        config2 = randi([-s s], 2, nPattern);
        ddd = cvtL2Norm([config1;config2], [config1;config2]);
        % 重複チェック
        if sum(ddd(:)==0) == nPattern
            flag = false;
        end
    end
    model.config1 = config1;
    model.config2 = config2;
    
    % 特徴抽出
    data_x = zeros(dim, nImg);
    for iImg = 1:nImg
        data_x(:,iImg) = cvtDBPP(imgs(:,:,iImg), model);
    end
    
    % 判別基準を最大に
    criterion = getFisherCriterion(data_x, data_l);
    model.criterion = criterion;
    if criterion > max_criterion
        max_criterion = criterion;
        max_model = model;
        max_data_x = data_x;
    end
end

end

function criterion = getFisherCriterion(data_x, data_l)

    nClass = max(data_l(:));
    Cc = cvtClassCov(data_x, data_l);
    cw = trace(sum(Cc,3));
    ct = trace(cov(data_x',1));
    criterion = cw / ct;

end