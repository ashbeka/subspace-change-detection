function  cvtPlotROC(EER,varargin)
FRR = EER.FRR1;
FAR = EER.FAR1;

if nargin < 1
   error('error');
end
if nargin == 1
    color = 'r';
end
if nargin == 2
    color = varargin{1};
end

hold on
plot(FAR,1-FRR,color);
title('ROC Curve');
xlabel('False Positive Rate');
ylabel('True  Positive Rate');
hold off

