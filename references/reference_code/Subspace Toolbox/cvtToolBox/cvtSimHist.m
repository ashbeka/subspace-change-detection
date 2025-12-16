function sim = cvtSimHist(hist_a, hist_b)

dim = size(hist_a, 1);
num_a = size(hist_a, 2);
num_b = size(hist_b, 2);

sim = zeros(num_a, num_b);


if num_a < num_b
    for bi = 1:num_b
        sim(:,bi) = sum(min(hist_a, repmat(hist_b(:,bi), 1, num_a)), 1);
    end
else
    for ai = 1:num_a
        sim(ai,:) = sum(min(repmat(hist_a(:,ai), 1, num_b), hist_b), 1);
    end
end

% A = repmat(hist_a(:), 1, m);
% B = repmat(hist_b   , n, 1);
%
% sim = sum(reshape(min(A, B), dim, n, m), 1);
%
% sim = squeeze(sim);

end
