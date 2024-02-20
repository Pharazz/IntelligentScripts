def sigmoid(input):
  tmp = (1/(1+np.exp(-input)))
  return tmp

def sigmoid_deriv(input):
  tmp = sigmoid(input)*(1-sigmoid(input))
  return tmp

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
  def __init__(self, value = 0, weights = [], bias = 0):
    self.value = value
    self.weights = weights
    self.bias = bias
    
  def clcVal(self, input):
    nxt = np.dot(self.weights, input)
    nxt += self.bias
    self.value = sigmoid(nxt)
  
  def getVal(self):
    return self.value
  
  def setWeights(self, newWeights):
    self.weights = newWeights
  
  def __str__(self):
    return "---\n Value: " + self.value + "\nWeights: {}".format(self.weights) + "\nBias: {}".format(self.bias)

def BatchGrad(theta_old, input, output, learningrate):
  Guess = sigmoid(input, theta_old) #Get guess for entire set
  Guess = np.squeeze(Guess) #Format
  theta_new = theta_old #Initialize new theta array shape
  for j in range(len(theta_old)):
    tmp = 0
    for i in range (len(input)):
      tmp += input[i][j]*(output[i] - Guess[i]) #Actual - Guess * Feature Input
    theta_new[j] = theta_old[j] + learningrate*(1/len(input))*(tmp) #Update Theta after all predictions have been logged
  return Guess, theta_new 

class NeuralNetwork:
  def __init__(self):
    values = 0
    weights = ([0,1])
    self.weights = weights
    bias = 0
    #Input layer is sent into class
    self.hidden1 = Neuron(values, weights, bias)
    self.hidden2 = Neuron(values, weights, bias)
    self.output1 = Neuron(values, weights, bias)
  
    self.out_error = 0
    self.out_delta = 0
    self.hidden_error = 0
    self.hidden_delta = 0
    
  def FeedFwd(self, input):
    self.hidden1.clcVal(input)
    self.hidden2.clcVal(input)
    self.output1.clcVal([self.hidden1.value,self.hidden2.value])
    return self.output1.getVal()
    
  def Backprop(self, input, labels):
    self.out_error = labels - self.output1.getVal()
    self.out_delta += self.out_error * sigmoid_derivative(self.output1.getVal())

    self.hidden_error = self.out_delta*(self.hidden1.weights)
    self.hidden_delta += self.hidden_error * sigmoid_derivative([self.hidden1.value,self.hidden2.value])
    return
    
  def UpdWeights(self):
    # Update weights
    self.weights[1] += [self.hidden1.value,self.hidden2.value].T.dot(self.out_delta) * 0.1
    self.weights[0] += input.T.dot(self.hidden_delta) * 0.1
    return
    
  def callMod(self, epochs, input, labels):
    for i in range(epochs):
      for k in range(len(input)):
        self.FeedFwd(input[k])
        self.Backprop(input[k], labels[k])
      self.UpdWeights()
    print(self.output1.getVal())
    return
  def makePred(self, input):
    pred = self.FeedFwd(input)
    print(pred)
    print(round(pred))
    return

Net = NeuralNetwork()
XOR = np.array([0,1,1,0])
testNet = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #np.array([0,1])

Net.callMod(10, testNet, XOR)
testy = np.array([0,1])
print(Net.FeedFwd(testy))
