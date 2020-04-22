function a_grad = activation_gradient(z);
  % finding the derivative for the activation function
  
  a_grad = sigmoid(z) * (1 - sigmoid(z));
  
endfunction