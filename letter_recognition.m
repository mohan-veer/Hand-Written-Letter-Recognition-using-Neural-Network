% Main File

clear; all clear ; clc;

% 3 layers
% 1 Input layer
% 1 Hidden layer
% 1 output layer

input_layer_size =  784; % 28*28 pixel image
output_layer_size = 26;  % a-z = 0-25 labels
hidden_layer_size = 400;  % https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
labels = 26; % 26 alphabets
% number of hidden layers is the mean of number of units present in input and hidden layer

% dataloading in X and y
X_train = [];
y_train = [];
X_test  = [];
y_test  = [];

[X_train,y_train,X_test,y_test] = dataloading();

fprintf('\n Data is Loaded\n');

%getting the parameters

initial_theta1 = network_weights_rand(input_layer_size,hidden_layer_size);
initial_theta2 = network_weights_rand(hidden_layer_size,output_layer_size);
initial_parameters = [initial_theta1(:) ; initial_theta2(:)];

fprintf('\n Intial Parameters have being initialized\n');
fprintf('\n Size of Initial parameters %d x %d\n',size(initial_parameters,1),size(initial_parameters,2));
% TRANING THE NEURAL NETWORK
% cost and partial derivatives

fprintf('\n TRAINING THE NEURAL NETWORK\n');
fprintf('Paused. Press enter to continue\n');
pause;

options = optimset('MaxIter',150);

lambda = 1;

costfunction = @(p) cost(p,X_train,y_train,input_layer_size,hidden_layer_size,labels,lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[final_parameters, J] = fmincg(costfunction, initial_parameters , options);
fprintf('\n MODEL IS TRAINED\n');
fprintf('\nThe cost of the model is : ');
display(J);
fprintf("\n");

% PREDICTING WITH THE TEST DATA

Theta1 = reshape(final_parameters(1:hidden_layer_size * (input_layer_size+1)),hidden_layer_size,input_layer_size+1);
Theta2 = reshape(final_parameters(hidden_layer_size * (input_layer_size+1)+1:end),labels,hidden_layer_size+1);

fprintf('\n Program paused. Press Enter to continue\n');
pause;

fprintf('\n PREDICTING WITH TRAIN DATA\n');
result_train = predict(Theta1,Theta2,X_train);
accuracy_train = mean(double(y_train==result_train));
accuracy_perct_train = accuracy_train * 100;
fprintf('\n The accuracy of the model with TRAINING DATA is %d\n',accuracy_perct_train);  


fprintf('\n PREDICTING WITH TEST DATA\n');
result = predict(Theta1,Theta2,X_test);


accuracy = mean(double(y_test==result));
accuracy_perct = accuracy * 100;
fprintf('\n The accuracy of the model with TESTING DATA is %d\n',accuracy_perct);  







