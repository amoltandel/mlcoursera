function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = length(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J1 = 0;
for i=1:1:m
    
    J1 = J1 + ((-1*y(i)*log( sigmoid(X(i,:)*theta ))) - ( 1-y(i) )*( log( 1-sigmoid(X(i,:) * theta))));
end

J1 = J1/m;
J2 = 0;
for j=2:1:n
    J2 = J2 + theta(j,1)^2;
end
J2 = J2 * lambda / (2 * m);
J = J1 + J2;
for j=1:1:n
    
    if j == 1
        
        for i=1:1:m
            grad(1,1)  = grad(1,1) + (sigmoid(X(i, :) * theta) - y(i)) * X(i, 1); 
        end
        grad(j,1) = grad(j,1)/m;
        
    else
        for i=1:1:m
            grad(j,1)  = grad(j,1) + (sigmoid(X(i, :) * theta) - y(i)) * X(i, j); 
        end
        grad(j,1) = (grad(j,1) + lambda * theta(j,1))/m;
        
        
    end
    
    
end

% =============================================================

end
