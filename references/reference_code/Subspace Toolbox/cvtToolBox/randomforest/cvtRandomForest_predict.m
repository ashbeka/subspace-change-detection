function [predict_l f_votes]= cvtRandomForest_predict(test_x, model, varargin)
%[predict_l f_votes]= cvtRandomForest_predict(test_x, model, varargin)
%Returns the output of the ensemble (f_output) as well
%as a [num_treesXnum_samples] matrix (f_votes) containing
%the outputs of the individual trees. 
%
%The 'oobe' flag allows the out-of-bag error to be used to 
%weight the final response (only for classification).

[predict_l f_votes]= eval_Stochastic_Bosque(test_x', model,varargin);
