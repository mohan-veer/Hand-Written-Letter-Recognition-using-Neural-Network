# Hand-Written-Letter-Recognition-using-Neural-Network
Implementation of Neural Network for Letter Recognition

* **Goal**: To implement a Neural network for recognizing the hand written alphabets.
 
**Dataset**: (https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)

* The dataset consists of pixel intensity values of images (hand written alphabets).
* The image size is **28*28 Pixel**. Thereby the feature count of the neural network is **28*28 = 784 columns**.
* Each column represents the pixel intensity value present at a particular position of pixel.
* Images constructed from pixel values present in dataset.

![Image 1 of dataset](/images/figure1.png)
![Image 2 of dataset](/images/figure2.png)

# MATH
* ***Logistic Regression-Sigmoid function*** is used as the hypothesis for classifying the english alphabets
* Minimizing the cost function-J(theta) is a major objective in order to achieve more efficient model.
* For minimizing the cost function **fmincg.m**, an advanced optimization method is used.
* A bias unit is added to both the Layer1 and Layer2
### Cost Function

![Cost Function](/images/cost.png)

### Sigmoid Function
* It Generates a value which is always between 0 and 1.
* 1 refers to a positive example and 0 to a negative example.
<div style="text-align:center">
<img src="/images/sigmoid.jpeg" /></div>

### Parameters
* There are two Parameter vectors each for one layer.(Theta1-Layer1,Theta2-Layer2)
* These parameter vectors are random intialized and then modified along with the iterations using **BACK PROPAGATION TECHNIQUE**

#### Epsilon value
* while random intialization an epsilon value is choosed for calculation.
* epsilon = sqrt(6)/(L_in + L_out), where L_in = number of units in layer l,L_out = number of units in layer l+1.

# Technical Overview

* A Neural network with 3 layers is considered in this scenario.
* The layers are:
1. **Input Layer**
2. **Hidden Layer**
3. **Output Layer**

![Neural Network](/images/neural_network.png)

* The number of units in the Input layer is equal to the feature size(28*28).
* The number of units in the Hidden layer should be in proportion with the input layer. One intuition can be referred form this website (https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
* The number of Output layer units is equal to the number of **labels** which are to be classified. So, number of units=26, as classification of English alphabets includes 26 labels.

## Activation Functions
* For layer 1 the activation includes the Input Data
* For Layer 2 the activation is ***Sigmoid Function***

## Data count
* The count of data examples for each label are inconsistent. If trained this way, model would be inefficient.
* Making the count of data examples consistent with respect to all labels.

![1-13 features count](/images/feature_count_1.png)
![14-26 features count](/images/feature_count_2.png)

* **Due to inconsistency in count of data rows, a common value is considered for all the labels i.e.,1100(for training and testing).**

## Cost function
* The count of the model after 150 iterations is:

<div style="text-align:center">
<img src="/images/cost_result.png" /></div>

## Accuracy
* **Accuracy with Test data is: 91.7%**
![Test data](/images/test.png)
* **Accuracy with Train data is: 98.3%**
![Test data](/images/train.png)
