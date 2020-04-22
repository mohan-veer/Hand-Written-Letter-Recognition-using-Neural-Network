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


% cost and partial derivatives




