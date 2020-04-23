% Main File

clear; all clear ; clc;

% 3 layers
% 1 Input layer
% 1 Hidden layer
% 1 output layer

input_layer_size =  784; % 28*28 pixel image
output_layer_size = 26;  % a-z = 0-25 labels
hiden_layer_size = 400;  % https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
labels = 26; % 26 alphabets
% number of hidden layers is the mean of number of units present in input and hidden layer

% dataloading in X and y

[X_train,y_train,X_test,y_test] = dataloading();

%getting the parameters

initial_theta1 = network_weights_rand(input_layer_size,hidden_layer_size);
initial_theta2 = network_weights_rand(hidden_layer_size,output_layer_size);
intial_parameters = [initial_theta1(:) ; initial_theta2(:)];

% TRANING THE NEURAL NETWORK
% cost and partial derivatives

options = optimset('MaxIter',100);
lambda = 1;
% we need to define a short hand for cost function, which is used to minimize the cost and genrate the gradient of weights

costfunction  = @(parameters) cost(parameters,X_train,y_train,labels,input_layer_size,hidden_layer_size,lambda);

[final_parameters,J] = fminunc(costfunction,initial_parameters,options);

% PREDICTING WITH THE TEST DATA

result = predict(Theta1,Theta2,X_test);

accuracy = mean(y_test==result);
accuracy_perct = accuracy * 100;
fprintf('\n The accuracy of the mode is %d',accuracy_perct);  







