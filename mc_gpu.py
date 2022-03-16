import numpy as np
import math
import torch
import os.path
import time

from numpy import random
from skimage import measure
from torch.autograd import Variable
from os import path

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: " + str(device))



class quaternion:
    x = 0.0;
    y = 0.0;
    z = 0.0;
    w = 0.0;

    def __init__(self, x=0, y=0, z=0, w=0):
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;

    def __assign__(self, value):
        self.x = value.x
        self.y = value.y
        self.z = value.z
        self.w = value.w

    def __str__(self): 
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.w);





num_components = 4; # quaternions
res = 25;

x_grid_max = 1.5;
y_grid_max = 1.5;
z_grid_max = 1.5;
x_grid_min = -x_grid_max;
y_grid_min = -y_grid_max;
z_grid_min = -z_grid_max;
x_res = res;
y_res = res;
z_res = res;
    
z_w = 0;

C = quaternion();
C.x = 0.3;
C.y = 0.5;
C.z = 0.4;
C.w = 0.2;

max_iterations = 8;
threshold = 4.0;
    
x_step_size = (x_grid_max - x_grid_min) / (x_res - 1);
y_step_size = (y_grid_max - y_grid_min) / (y_res - 1);
z_step_size = (z_grid_max - z_grid_min) / (z_res - 1);

Z = quaternion();
Z.x = x_grid_min;
Z.y = y_grid_min;
Z.z = z_grid_min;
Z.w = z_w;




float_slice_a = torch.zeros((res* res, num_components), dtype=torch.float32)#.to(device);
float_slice_b = torch.zeros((res* res, num_components), dtype=torch.float32)#.to(device);
float_array = np.zeros((res, res, res), dtype = np.float32);


def get_predictions():
    batch = torch.cat((float_slice_a, float_slice_b), 1)
    #batch.to(device)
    return net(batch).detach().numpy();



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





net = Net()#.to(device)

if path.exists('weights_4_100000.pth'):
    net.load_state_dict(torch.load('weights_4_100000.pth'))
    print("loaded file successfully")
else:
    print("Could not find file...")
    exit()
          




t0= time.perf_counter()




x_steps = torch.linspace(x_grid_min, x_grid_max, steps=x_res);
y_steps = torch.linspace(y_grid_min, y_grid_max, steps=y_res);
z_steps = torch.linspace(z_grid_min, z_grid_max, steps=z_res);

#loc = np.zeros((res, res, res), dtype = quaternion);

loc = np.zeros((res * res * res), dtype = quaternion);

for z in range(res):#z_step in z_steps:
    for x in range(res):#x_step in x_steps:
        for y in range(res):#y_step in y_steps:
            loc[z*res*res + y*res + x] = quaternion(x_steps[x], y_steps[y], z_steps[z], z_w);





for i in range(z_res):

    print(str(i))

    print("init")
    
    for j in range(x_res):

        for k in range(y_res):
        
            index = j*res + k;

            Z = loc[i*res*res + k*res + j];# = quaternion(x_steps[x], y_steps[y], z_steps[z], z_w);

            float_slice_a[index][0] = Z.x;
            float_slice_a[index][1] = Z.y;
            float_slice_a[index][2] = Z.z;
            float_slice_a[index][3] = Z.w;
            float_slice_b[index][0] = Z.x;
            float_slice_b[index][1] = Z.y;
            float_slice_b[index][2] = Z.z;
            float_slice_b[index][3] = Z.w;

        #    Z.y += y_step_size;

       # Z.x += x_step_size;
   
    #float_slice_a.to(device);
    #float_slice_b.to(device);


    print("done init")


    float_slice_magnitude = np.zeros((res, res), dtype=np.float32);

    for m in range(max_iterations):

#        print(m);

        p = get_predictions();

        Z.x = x_grid_min;

        for j in range(x_res):
            Z.y = y_grid_min;

            for k in range(y_res):

                index = j*res + k

                p[index][0] += C.x;
                p[index][1] += C.y;
                p[index][2] += C.z;
                p[index][3] += C.w;

                float_slice_magnitude[j][k] = math.sqrt(p[index][0]*p[index][0] + p[index][1]*p[index][1] + p[index][2]*p[index][2] + p[index][3]*p[index][3]);
                float_slice_a[index][0] = float_slice_b[index][0] = torch.from_numpy(p)[index][0];
                float_slice_a[index][1] = float_slice_b[index][1] = torch.from_numpy(p)[index][1];
                float_slice_a[index][2] = float_slice_b[index][2] = torch.from_numpy(p)[index][2];
                float_slice_a[index][3] = float_slice_b[index][3] = torch.from_numpy(p)[index][3];

                Z.y += y_step_size;

            Z.x += x_step_size;

        #float_slice_a.to(device);
        #float_slice_b.to(device);

    Z.z += z_step_size;

    float_array[i] = float_slice_magnitude;
"""
t1 = time.perf_counter()

print("Time elapsed: ", t1 - t0)


verts, faces, normals, values = measure.marching_cubes(float_array, threshold)

thefile = open('test_ai.obj', 'w')
for item in verts:
  thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

for item in normals:
  thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

for item in faces:
  thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0] + 1, item[1] + 1, item[2] + 1))  

thefile.close()



