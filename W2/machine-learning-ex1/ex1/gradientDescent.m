function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp1 = X * theta;
    temp1 = temp1 - y;% 97x1
    s1 = sum(temp1);
    change1 = alpha * s1 / m;

    temp2 = temp1 .* X(:, 2);
    s2 = sum(temp2);
    change2 = alpha * s2 / m;

    theta(1, 1) = theta(1, 1) - change1;
    theta(2, 1) = theta(2, 1) - change2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
