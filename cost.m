function [J grad] = cost(parameters,X_train,y_train,input_layer_size,hidden_layer_size,labels,lambda)


% converting parameters to indvidual THETA values

Theta1 = reshape(parameters(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size,(input_layer_size + 1));

Theta2 = reshape(parameters((1 + (hidden_layer_size * (input_layer_size + 1))):end),labels,(hidden_layer_size + 1));

theta_grad1 = zeros(size(Theta1));
theta_grad2 = zeros(size(Theta2));
% deriving the cost of the model and grad vectors of parameters
K = labels;
J = 0;
reg_term = 0;
m = size(X_train,1);

y_new = [1:K == y_train];

a1 = X_train;
a1 = [ones(size(a1),1) a1];  % adding a bias unit for layer 1

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2),1) a2];  % adding bias unit for layer 2

z3 = a2 * Theta2';
a3 = sigmoid(z3);    % HYPOTHESIS OF THE MODEL

% cost function

temp = ((-y_new) .* log(a3)) - ((1-y_new) .* (log(1-a3)));
temp = sum(sum(temp,2));
J = J + (1/m) * temp;
 

 
 % regualrization term
 reg_term = (lambda/(2*m)) * [sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))];
 
 
 J = J + reg_term;
 
 % back propagation
 
d3 = a3 - y_new;

activation_derivative_result = activation_derivative(z2);
activation_derivative_result = [ones(size(activation_derivative_result,1),1) activation_derivative_result];
d2 = (d3 * Theta2) .* activation_derivative_result;
d2 = d2(:,2:end);

theta_grad1 = theta_grad1 + (d2)' * a1;
theta_grad2 = theta_grad2 + (d3)' * a2;


theta_grad1 = theta_grad1 * (1/m);
theta_grad2 = theta_grad2 * (1/m);

theta_grad1(:,2:end) = theta_grad1(:,2:end) + (lambda/m) * Theta1(:,2:end);
theta_grad2(:,2:end) = theta_grad2(:,2:end) + (lambda/m) * Theta2(:,2:end);

grad = [theta_grad1(:) ; theta_grad2(:)];
endfunction



   
   
   
 
