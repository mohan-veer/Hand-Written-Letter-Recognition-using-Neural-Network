function [J,grad] = cost(X_train,y_train,parameters,labels,input_layer_size,hidden_layer_size,lambda)

% deriving the cost of the model and grad vectors of parameters
K = labels;
J = 0;
reg_term = 0;
m = size(X_train,1);

y_new = zeros(m,labels);
for i=1:m,
  y_new(i,y_train(i)) = 1;
endfor

a1 = X_train;
a1 = [ones(size(a1),1) a1];  % adding a bias unit for layer 1

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2),1) a2];  % adding bias unit for layer 2

z3 = a2 * Theta2';
a3 = sigmoid(z3);    % HYPOTHESIS OF THE MODEL

% cost function

for i=1:m,
  for k=1:K,
    J = J + [(y_new(i,k) * log(a3(i,k))) + ((1-y_new(i,k)) - log(1-a3(i,k)))];
  endfor
 endfor
 
 % regualrization term
 reg_term = (lambda/(2*m)) * [sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))];
 
 
 J = (1/m) * J + reg_term;
 
 % back propagation
 
 for i:1:m,
   a1 = X(1,:);
   a1 = [ones(size(a1),1) a1];
   
   z2 = a1 * Theta1';
   a2 = sigmoid(z2);
   a2 = [ones(size(a2),1) a2];
   
   z3 = a2 * Theta2';
   a3 = sigmoid(z3);
   a3 = a3';
   
   y_n = [1:K == y_train(i)]';
   
   d3 = a3 - y_n;
   
   d2 = a2' * d3 .* [1;activation_derivative(z2)];
   d2 = d2(:,2:end);
   
   theta_grad1 = theta_grad1 + d2 * (a1)';
   theta_grad2 = theta_grad2 + d3 * (a2)';
   
endfor

theta_grad1 = theta_grad1 * (1/m);
theta_grad2 = theta_grad2 * (1/m);

theta_grad1(:,2:end) = theta_grad1(:,2:end) + (lambda/m) * Theta1(:,2:end);
theta_grad2(:,2:end) = theta_grad2(:,2:end) + (lambda/m) * Theta2(:,2:end);


endfunction



   
   
   
 
