% loading data 
function [X_train,y_train,X_test,y_test] = dataloading()

% data set consists of 28*28 pixel image of letters from (a-z)
%total features = 784
%labels = 0 to 25(a-z)
%number of training examples = 372451
labels = 26;
total_data = csvread("handwritten.csv");
% Number of trainig examples are too large to handle 
% we will select few random examples for the model
% labeling is  from 0-25 :  changing them to 1-26 help us in easy indexing
total_data(:,1) = total_data(:,1)+1;
fprintf('\nchecking whether labels are updated or not\n')
total_data(1,1)
total_data(268,1)
total_data(372451,1)

% allocating training and testing values with a 75% data to train and 25% data to test
X_train = [];
y_train = [];
X_test = [];
y_test = [];

% setting 100 samples from each laebl for testing
for i=1:labels,
  temp_data = total_data(total_data(:,1)==i,:);
  total = size(temp_data,1);
  train = 1000;
  fprintf('\nTotal rows with label %d are : %d\n',i,total);
  X_train = [X_train;temp_data(1:train,2:end)];
  X_test = [X_test;temp_data(train+1:train+100,2:end)]; 
  y_train = [y_train;temp_data(1:train,1)];
  y_test = [y_test;temp_data(train+1:train+100,1)];
endfor

%X_train = total_data(:,2:end);
%y_train = total_data(:,1);

fprintf('\n The size of training features : %d x %d ',size(X_train,1),size(X_train,2));
fprintf('\n The size of training labels : %d x %d',size(y_train,1),size(y_train,2));

fprintf('\n The size of testing features : %d x %d ',size(X_test,1),size(X_test,2));
fprintf('\n The size of testing features : %d x %d ',size(y_test,1),size(y_test,2));

rand_numbers = [56;223;900]
for i=1:3,
  fprintf('\nImage of alphabet with row number %d is displayed\n',i);
  v = rand_numbers(i);
  image = X_train(v,:)';
  image = reshape(image,28,28);
  imshow(image);
  pause;
endfor

endfunction