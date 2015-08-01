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
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X_t = [ones(m, 1) X]; % m x 401
z2 = Theta1 * X_t'; % 25 x m
a2 = sigmoid(z2); % 25 x m
a2_t = [ones(1, m); a2]; % 26 x m
z3 = Theta2 * a2_t; % 10 x m
a3 = sigmoid(z3); % 10 x m
y_t = zeros(num_labels, m);
for i = 1:m,
    y_t(y(i), i) = 1;
end
J = -1.0 / m * sum(sum(y_t .* log(a3) .+ (1 .- y_t) .* log(1 .- a3)));
reg = lambda / 2.0 / m * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + reg;

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
%

for t = 1:m,
    delta3 = a3(:, t) .- y_t(:, t); % 10 x 1
    delta2 = Theta2' * delta3; 
    delta2 = delta2(2:end); % 25 x 1
    delta2 = delta2 .* sigmoidGradient(z2(:, t)); % 25 x 1


    Theta1_grad = Theta1_grad .+ delta2 * X_t(t, :);
    Theta2_grad = Theta2_grad .+ delta3 * a2_t(:, t)';
end
Theta1_grad /= m;
Theta2_grad /= m;

for j = 1:hidden_layer_size,
    for k = 2:input_layer_size+1
        Theta1_grad(j, k) += Theta1(j, k) * lambda / m;
    end
end
for j = 1:num_labels,
    for k = 2:hidden_layer_size+1
        Theta2_grad(j, k) += Theta2(j, k) * lambda / m;
    end
end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
