function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_mod = zeros(m, num_labels);
for i=1:1:m
    y_mod(i, y(i)) = 1;
end

for i=1:1:m
    a1 = [1; X(i, :)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    for k=1:1:num_labels
        J = J + (-1 * y_mod(i, k) * log(a3(k)) - (1 - y_mod(i, k)) * log(1-a3(k)));
    end
end

J = J/m;

temp = sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2));

J = J + (lambda * temp / (2*m));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta2 = zeros(num_labels, hidden_layer_size + 1);
for i=1:1:m
    
    a1 = [1; X(i, :)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    delta3 = a3 - y_mod(i, :)';
    
    
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    
    Delta2 = Delta2 + delta3 * a2';
    
    Delta1 = Delta1 + delta2 * a1';
    
    
end

Delta2 = Delta2/m;
Delta1 = Delta1/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%             a1 = X(i, :);
      
%         the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

temp1 = lambda * [zeros(hidden_layer_size, 1) Theta1(:, 2:end)] / m;

temp2 = lambda * [zeros(num_labels, 1) Theta2(:, 2:end)] / m;


Delta1 = Delta1 + temp1;
Delta2 = Delta2 + temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Delta1(:) ; Delta2(:)];


% %Pt2
% Del_1 = zeros(hidden_layer_size, input_layer_size + 1);
% Del_2 = zeros(num_labels, hidden_layer_size);
% %disp(size(Del_1));
% for i=1:1:m
%     a1 = X(i, :)';
%     z2 = Theta1 * a1;
%     a2 = sigmoid(z2);
%     z3 = [1 ; a2];
%     z3 = Theta2 * z3;
%     a3 = sigmoid(z3);
%     
%     delta_3 = a3 - y_mod(i, :)';
%     
% %    size(delta_3)
% %    delta_3 = [1; delta_3];
% %     disp(size(delta_3));
% %     disp(size(Theta2'));
% %     disp(size([1; sigmoidGradient(z2)]));
%     delta_2 = Theta2' * delta_3 .* [1; sigmoidGradient(z2)];
%     
%     delta_2 = delta_2(2:end);
%     %disp(size(delta_2 * a1'));
%     Del_2 = Del_2 + delta_3 * a2';
%     
%     Del_1 = Del_1 + delta_2 * a1';
%     
% end
% 
% grad = [Del_1(:) ; Del_2(:)];
% grad = grad/m;


end
