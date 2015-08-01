function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
h = sigmoid(X * theta);
theta_t = theta(2:end);
J = sum(-y .* log(h) - (1.0 - y) .* log(1.0 - h)) / m + lambda / 2.0 / m * sum(theta_t.^2);
grad = (h - y)' * X / m;
grad(2:end) = grad(2:end) + lambda / m .* theta_t';




% =============================================================

end
