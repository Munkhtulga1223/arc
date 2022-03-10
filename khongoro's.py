def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    
    filter_du = np.stack([filter_du] * 222, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    convolved = np.sqrt(np.square(convolve(img, filter_du)) + np.square(convolve(img, filter_dv)))

    energy_map = convolved.sum(axis=2)
    return energy_map
    
    print('zuraggui bol ur dungui to make a conflict') 

import numpy as np

def sigmoid(x):
  # Идэвхжүүлэх Сигмоид функц: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # сигмоидын дифференциал олох: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true болон y_pred - массивуудийг numpy сан ашиглан ижил урттай тодорхойлох:
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  Нейроны сүлжээ:
    - 2 оролттой
    - 2 нейрон (h1, h2) бүхий 1 далд давхаргатай 
    - 1 нейрон бүхий гаралтын давхарга (o1) '''

  def __init__(self):
    # жин
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # bias
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # гаралтууд
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    
    learn_rate = 0.1
    epochs = 1000  

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        
        # --- урагш тархалт буюу feedforward
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- дифференциал бодох.
        
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Нейрон h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Нейрон h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Жин, биасын утгыг шинэчилж байна 
        # Нейрон h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Нейрон o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Эпох бүрийн төгсгөлд бид нийт алдагдлыг тооцно
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Өгөгдлийн багц буюу датасет тодорхойлох
data = np.array([
  [4, 2],  # Болд
  [-6, -4],   # Оюу
  [10, 11],   # Бат
  [-10, -7], # Сувд
  [-6, -2], # Eej
  [15,0], #aav
])
all_y_trues = np.array([
  0, # Болд
  1, # Оюу
  0, # Бат
  1, # Сувд
  1, # Eej
  0, # Aav 
])

# Сүлжээг сургах:
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Таамаглал дэвшүүлэх:
Khulan = np.array([-4, 1]) #  (60 кг), (170 см)
Dorj = np.array([16, 6])  # 155 pounds (80 кг), (175 см)
Aav = np.array([15, 0])  # (79kg), (169 cm)
Eej = np.array([-6, -2]) # (58kg), (167cm)


print("Хулан: %.3f" % network.feedforward(Khulan)) # 0.947 - эм
print("Дорж: %.3f" % network.feedforward(Dorj)) # 0.055 - эр
print("Aav: %.3f" % network.feedforward(Aav)) 
print("Eej: %.3f" % network.feedforward(Eej)) 