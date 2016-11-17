function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% C_list = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
% sigma_list = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
% C_best = 0;
% sigma_best = 0;
% err_best = 1000000;
% for i=1:1:9
%     for j=1:1:9
%         
%         C = C_list(i);
%         sigma = sigma_list(j);
%         model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%         preds = svmPredict(model, Xval);
%         err = mean(double(preds ~= yval));
%         
%         if err < err_best
%             err_best = err;
%             C_best = C;
%             sigma_best = sigma;
%         end
%         
%     end
% end
C = 1;
sigma = 0.1;

% =========================================================================

end
