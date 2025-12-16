function [predicted_label, accuracy, decision_values_prob_estimates] = cvtSVM_predict(model, test_x, test_l)

[predicted_label, accuracy, decision_values_prob_estimates] = ...
    predict(test_l', sparse(test_x'), model);

end
