% test cvtSimHist

N = 100;
n = 60;
m = 40;
dim = 10;
hist_a = zeros(dim, n);
hist_b = zeros(dim, m);

for ni = 1:n;
   hist_a(:,ni) = hist(mod(randn(100,1)*0.1+0.1*ni,1), 0.1:0.1:1); 
end

for mi = 1:m;
   hist_b(:,mi) = hist(mod(randn(100,1)*0.1+0.1*mi,1), 0.1:0.1:1); 
end
sim = cvtSimHist(hist_a, hist_b);

imagesc(sim);

%%
for i = 1:100
    sim = cvtSimHist(hist_a, hist_b);
end
