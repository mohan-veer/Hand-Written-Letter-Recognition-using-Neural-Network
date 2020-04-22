% loading data 
function [X_train,y_train,X_test,y_test] = dataloading()

% data set consists of 28*28 pixel image of letters from (a-z)
%total features = 784
%labels = 0 to 25(a-z)
%number of training examples = 372451

total_data = csvread("handwritten.csv");
% Number of trainig examples are too large to handle 
% we will select few rando examples for the model

data = datasample(total_data,13000);
X_train = data(:,2:end);
y_train = data(:,1);

data_testing = datasample(total_data,1000);
X_test = data_testing(:,2:end);
y_test = data_testing(:,1);

% labeling is  from 0-25 :  changing them to 126 help us in easy indexing
y_train = y_train + 1;
y_test = y_test + 1;

fprintf('\n The size of features : ',size(X));
fprintf('\n The size of labels : ',size(y));

endfunction