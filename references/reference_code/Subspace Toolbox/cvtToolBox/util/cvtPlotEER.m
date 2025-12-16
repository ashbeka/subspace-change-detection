function  cvtPlotEER(EER)

FRR = EER.FRR1;
FAR = EER.FAR1;
A   = EER.A1;
clf;
hold on
plot(A,FRR,'b');
plot(A,FAR,'r');
title('FRR - FAR');
legend('False Reject Rate','False Alarm  Rate',0);
xlabel('Threshold');
ylabel('Rate');
hold off

