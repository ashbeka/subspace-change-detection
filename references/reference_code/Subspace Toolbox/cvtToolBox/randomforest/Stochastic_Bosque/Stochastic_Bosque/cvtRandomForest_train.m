function Random_Forest = cvtRandomForest_train(train_x, train_l, varargin)
%Random_Forest = Stochastic_Bosque(Data,Labels,varargin)
%
%   Creates an ensemble of CARTrees using Data(samplesXfeatures).
%   The following parameters can be set :
%
%       ntrees       : number of trees in the ensemble (default 50)
%
%       oobe         : out-of-bag error calculation, 
%                      values ('y'/'n' -> yes/no) (default 'n')
%
%       nsamtosample : number of randomly selected (with
%                      replacement) samples to use to grow
%                      each tree (default num_samples)
%
%
%   Furthermore the following parameters can be set regarding the
%   trees themselves :
%
%       method       : the criterion used for splitting the nodes
%                           'g' : gini impurity index (classification)
%                           'c' : information gain (classification)
%                           'r' : squared error (regression)
%
%       minparent    : the minimum amount of samples in an impure node
%                      for it to be considered for splitting
%
%       minleaf      : the minimum amount of samples in a leaf
%
%       weights      : a vector of values which weigh the samples 
%                      when considering a split
%
%       nvartosample : the number of (randomly selected) variables 
%                      to consider at each node 

Random_Forest = Stochastic_Bosque(train_x', train_l, varargin);

