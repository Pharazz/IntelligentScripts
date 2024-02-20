class Struc1RELU(Model): #Change this from Model to Keras.Sequential instantiation with the layers inside of it, or make a compile method
  def __init__(self):
    super(Struc1RELU, self).__init__()
    self.d1 = Dense(12, activation='relu', input_shape=(1,))
    self.d2 = Dense(8, activation='relu')
    self.d3 = Dense(4, activation = 'relu')
    self.d4 = Dense(1)
    self.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)

class Struc1TANH(Model):
  def __init__(self):
    super(Struc1TANH, self).__init__()
    self.d1 = Dense(12, activation='tanh')
    self.d2 = Dense(8, activation='tanh')
    self.d3 = Dense(4, activation = 'tanh')
    self.d4 = Dense(1)
    self.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)

class Struc2RELU(Model):
  def __init__(self):
    super(Struc2RELU, self).__init__()
    self.d1 = Dense(24, activation='relu')
    self.d2 = Dense(1)
    self.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])
  def call(self, x):
    x = self.d1(x)
    return self.d2(x)

class Struc2TANH(Model):
  def __init__(self):
    super(Struc2TANH, self).__init__()
    self.d1 = Dense(24, activation='tanh')
    self.d2 = Dense(1)
    self.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])
  def call(self, x):
    x = self.d1(x)
    return self.d2(x)
