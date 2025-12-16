function [VOTE M] = cvtVoting(SIM)

[nClass nNum nVote] = size(SIM);
VOTE = zeros(nClass,nNum);
if nVote > 1
    [~, ind] = max(SIM,[],1);
    ind = squeeze(ind);
    for I=1:nNum
        for J=1:nVote
            VOTE(ind(I,J),I) = VOTE(ind(I,J),I) + 1;
        end
    end
    M = mean(SIM,3);
elseif nVote ==1
    [~, ind] = max(SIM,[],1);
    for I=1:nNum
        VOTE(ind(I),I) = 1;
    end
    M = SIM;
end
