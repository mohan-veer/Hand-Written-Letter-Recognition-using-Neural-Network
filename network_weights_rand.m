% generating the weights for the neural network
function Theta = network_weights_rand(L_in,L_out)
  
  % generating epsilon value for each theta separately based on the formula
  % epsilon = sqrt(6)/sqrt(L_in + L_out) 
  % L_in = s_l (number of units present in layer-l) 
  % L_out = s_l+1 (number of units prsent in layer-l+1)
  
  epsilon = sqrt(6)/(sqrt(L_in+L_out));
  Theta = rand(L_out,L_in+1) * (2*epsilon) - epsilon;
  
endfunction
