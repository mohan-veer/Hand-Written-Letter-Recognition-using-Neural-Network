% PREDICTING WIHT THE MODEL
function results = predict(Theta1,Theta2,X_test)
  a1 = X_test;
  a1 = [ones(size(a1),1) a1];  % adding a bias unit for layer 1

  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2),1) a2];  % adding bias unit for layer 2

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);    % HYPOTHESIS OF THE MODEL
  
  [value,index] = max(a3,[],2);
  results = index;
endfunction
