import numpy as np
import math
import torch
from torch.autograd import Variable

import numpy as np
import time



num_components = 4;
    


def traditional_mul_replacement(in_a, in_b):

  out = np.zeros([num_components], np.float32)
  answer = 0
  for i in range(num_components):
    for j in range(num_components):
      answer += in_a[i] * in_b[j];

  return out.T;

"""
	out[0]= in_a[0]* in_b[0]- in_a[1]* in_b[1]- in_a[2]* in_b[2]- in_a[3]* in_b[3]- in_a[4]* in_b[4]- in_a[5]* in_b[5]- in_a[6]* in_b[6]- in_a[7]* in_b[7];
	out[1]= in_a[0]* in_b[1]+ in_a[1]* in_b[0]+ in_a[2]* in_b[3]- in_a[3]* in_b[2]+ in_a[4]* in_b[5]- in_a[5]* in_b[4]- in_a[6]* in_b[7]+ in_a[7]* in_b[6];
	out[2]= in_a[0]* in_b[2]- in_a[1]* in_b[3]+ in_a[2]* in_b[0]+ in_a[3]* in_b[1]+ in_a[4]* in_b[6]+ in_a[5]* in_b[7]- in_a[6]* in_b[4]- in_a[7]* in_b[5];
	out[3]= in_a[0]* in_b[3]+ in_a[1]* in_b[2]- in_a[2]* in_b[1]+ in_a[3]* in_b[0]+ in_a[4]* in_b[7]- in_a[5]* in_b[6]+ in_a[6]* in_b[5]- in_a[7]* in_b[4];
	out[4]= in_a[0]* in_b[4]- in_a[1]* in_b[5]- in_a[2]* in_b[6]- in_a[3]* in_b[7]+ in_a[4]* in_b[0]+ in_a[5]* in_b[1]+ in_a[6]* in_b[2]+ in_a[7]* in_b[3];
	out[5]= in_a[0]* in_b[5]+ in_a[1]* in_b[4]- in_a[2]* in_b[7]+ in_a[3]* in_b[6]- in_a[4]* in_b[1]+ in_a[5]* in_b[0]- in_a[6]* in_b[3]+ in_a[7]* in_b[2];
	out[6]= in_a[0]* in_b[6]+ in_a[1]* in_b[7]+ in_a[2]* in_b[4]- in_a[3]* in_b[5]- in_a[4]* in_b[2]+ in_a[5]* in_b[3]+ in_a[6]* in_b[0]- in_a[7]* in_b[1];
	out[7]= in_a[0]* in_b[7]- in_a[1]* in_b[6]+ in_a[2]* in_b[5]+ in_a[3]* in_b[4]- in_a[4]* in_b[3]- in_a[5]* in_b[2]+ in_a[6]* in_b[1]+ in_a[7]* in_b[0];
"""




def ground_truth(batch):
  truth = np.zeros([batch.shape[0],num_components],np.float32);
  for i in range(batch.shape[0]):
    a = batch[i,0:num_components]
    b = batch[i,num_components:num_components*2]
    truth[i,:] = traditional_mul_replacement(a,b);
  return truth;

def normalize_batch(batch):

  for i in range(batch.shape[0]):

    batch[i, 0:num_components] /= math.sqrt(np.dot(batch[i, 0:num_components], batch[i, 0:num_components]));
    batch[i, num_components:num_components*2] /= math.sqrt(np.dot(batch[i, num_components:num_components*2], batch[i, num_components:num_components*2]));

  return batch;


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(num_components*2, 32*num_components)
        self.hidden2 = torch.nn.Linear(32*num_components, 16*num_components) 
        self.hidden3 = torch.nn.Linear(16*num_components, 8*num_components)
        self.predict = torch.nn.Linear(8*num_components, num_components)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))      
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x



net = Net()

#print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
loss_func = torch.nn.MSELoss()


for epoch in range(100000):

  batch = torch.rand((100,num_components*2),dtype=torch.float32) * 2 - 1
  batch = normalize_batch(batch)

  gt = ground_truth(batch.numpy())
  x = Variable(batch)
  y = Variable(torch.from_numpy(gt))

  prediction = net(x)     
  loss = loss_func(prediction, y)
  if epoch % 500 == 0:
    print(epoch,loss)
  optimizer.zero_grad()   # clear gradients for next train
  loss.backward()         # backpropagation, compute gradients
  optimizer.step()        # apply gradients

torch.save(net.state_dict(), 'model_weights' + str(num_components) + '.pth')



t0= time.perf_counter()

for i in range(10000):

    if i % 500 == 0:
        print(i)

    batch = torch.rand((10,num_components*2),dtype=torch.float32) * 2 - 1
    batch = normalize_batch(batch)

    gt = ground_truth(batch.numpy())

t1 = time.perf_counter()

print("Time elapsed: ", t1 - t0)



t0= time.perf_counter()

for i in range(10000):

    if i % 500 == 0:
        print(i)

    batch = torch.rand((10,num_components*2),dtype=torch.float32) * 2 - 1
    batch = normalize_batch(batch)

    prediction = net(batch).detach().numpy()
    #loss = loss_func(prediction, y)

    prediction = normalize_batch(prediction)

t1 = time.perf_counter()

print("Time elapsed: ", t1 - t0)


"""
print(batch)

print("\n")

print(gt)

print("\n")

print(prediction)
"""

