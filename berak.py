import numpy as np
import math
import torch
from torch.autograd import Variable


    import os.path
    from os import path



num_components = 4
threshold = 4.0
num_epochs = 100000


def traditional_mul(in_a, in_b):

  out = np.zeros([num_components], np.float32)

  out[0]= in_a[0]* in_b[0]- in_a[1]* in_b[1]- in_a[2]* in_b[2]- in_a[3]* in_b[3]
  out[1]= in_a[0]* in_b[1]+ in_a[1]* in_b[0]+ in_a[2]* in_b[3]- in_a[3]* in_b[2]
  out[2]= in_a[0]* in_b[2]- in_a[1]* in_b[3]+ in_a[2]* in_b[0]+ in_a[3]* in_b[1]
  out[3]= in_a[0]* in_b[3]+ in_a[1]* in_b[2]- in_a[2]* in_b[1]+ in_a[3]* in_b[0]
    
  return out.T;


def ground_truth(batch):
  truth = np.zeros([batch.shape[0],num_components],np.float32);
  for i in range(batch.shape[0]):
    a = batch[i,0:num_components]
    b = batch[i,num_components:num_components*2]
    truth[i,:] = traditional_mul(a,b);
  return truth;

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

if path.exists('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'):
    net.load_state_dict(torch.load('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'))
    print("loaded file successfully")
else:
    print("training...")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):

      batch = threshold * (torch.rand((100,num_components*2),dtype=torch.float32) * 2 - 1)

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

    torch.save(net.state_dict(), 'weights_' + str(num_components) + '_' + str(num_epochs) + '.pth')


batch = threshold*(torch.rand((10, num_components*2),dtype=torch.float32) * 2 - 1)
gt = ground_truth(batch.numpy())
prediction = net(batch).detach().numpy()

print(gt)
print("\n")
print(prediction)


