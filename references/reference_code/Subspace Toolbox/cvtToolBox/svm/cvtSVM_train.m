function model = cvtSVM_train(train_x, train_l, option)
% -training_label_vector:
%     An m by 1 vector of training labels. (type must be double)
% -training_instance_matrix:
%     An m by n matrix of m training instances with n features.
%     It must be a sparse matrix. (type must be double)
% -liblinear_options:
%     A string of training options in the same format as that of LIBLINEAR.
% -col:
%     if 'col' is set, each column of training_instance_matrix is a data instance. Otherwise each row is a data instance.          
%         
% liblinear_options:
% -s type : set type of solver (default 1)
% 	0 -- L2-regularized logistic regression (primal)
% 	1 -- L2-regularized L2-loss support vector classification (dual)
% 	2 -- L2-regularized L2-loss support vector classification (primal)
% 	3 -- L2-regularized L1-loss support vector classification (dual)
% 	4 -- multi-class support vector classification by Crammer and Singer
% 	5 -- L1-regularized L2-loss support vector classification
% 	6 -- L1-regularized logistic regression
% 	7 -- L2-regularized logistic regression (dual)
% -c cost : set the parameter C (default 1)
% -e epsilon : set tolerance of termination criterion
% 	-s 0 and 2
% 		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
% 		where f is the primal function and pos/neg are # of
% 		positive/negative data (default 0.01)
% 	-s 1, 3, 4 and 7
% 		Dual maximal violation <= eps; similar to libsvm (default 0.1)
% 	-s 5 and 6
% 		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
% 		where f is the primal function (default 0.01)
% -B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
% -wi weight: weights adjust the parameter C of different classes (see README for details)
% -v n: n-fold cross validation mode
% -q : quiet mode (no outputs)

if nargin < 3
    option = [];
end


if numel(unique(train_l)) == 2
% 2 class classification
%     model = liblinear_train(train_l', train_x');
    model = train(train_l', sparse(train_x)', [option '-B -1']);
else
% multi-class classification
    model = train(train_l', sparse(train_x)', [option]);
end
% disp(option);




end
