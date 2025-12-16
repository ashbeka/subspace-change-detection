function C = cvtSimBhattacharyya(SIM,L,band)

pSIM = [];
nSIM = [];
for H=1:size(SIM,1)
   pSIM = [pSIM,SIM(H,H==L)];
   tmpSIM = [SIM(1:size(SIM,1)~=H,H==L)];
   nSIM = [nSIM,tmpSIM(:)'];
end
pa  = hist(nSIM,band);
na  = hist(pSIM,band);
C = sum(sqrt((pa/sum(pa)).*na/sum(na)));
% 
% figure(234234)
% clf
% hold on
% plot(band,pa,'r')
% plot(band,na,'b')
% hold off 
% 
