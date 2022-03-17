import numpy as np
import math
import torch
import os.path
import time
import numba

from numpy import random
from skimage import measure
from torch.autograd import Variable
from os import path


device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


num_components = 4; # quaternions


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

    def __str__(self): 
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.w);




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

@numba.jit
def calc_slice(float_slice, grid_min, res, Z, C, step_size, p, float_array, i):

    Z.x = grid_min;

    for j in range(res):

       Z.y = grid_min;

       for k in range(res):

           index = j*res + k

           p[index][0] += C.x;
           p[index][1] += C.y;
           p[index][2] += C.z;
           p[index][3] += C.w;

           float_array[i][j][k] = math.sqrt(p[index][0]*p[index][0] + p[index][1]*p[index][1] + p[index][2]*p[index][2] + p[index][3]*p[index][3]);
   
           tp = torch.from_numpy(p);

           float_slice[index][0] = tp[index][0];
           float_slice[index][1] = tp[index][1];
           float_slice[index][2] = tp[index][2];
           float_slice[index][3] = tp[index][3];
           float_slice[index][4] = tp[index][0];
           float_slice[index][5] = tp[index][1];
           float_slice[index][6] = tp[index][2];
           float_slice[index][7] = tp[index][3];
  
           Z.y += step_size;

       Z.x += step_size;





@numba.jit
def init_slice(float_slice, grid_min, res, Z, step_size):

    Z.x = grid_min;

    for j in range(res):

        Z.y = grid_min;

        for k in range(res):
        
            index = j*res + k;

            float_slice[index][0] = Z.x;
            float_slice[index][1] = Z.y;
            float_slice[index][2] = Z.z;
            float_slice[index][3] = Z.w;
            float_slice[index][4] = Z.x;
            float_slice[index][5] = Z.y;
            float_slice[index][6] = Z.z;
            float_slice[index][7] = Z.w;

            Z.y += step_size;

        Z.x += step_size;


def main():

    res = 100;
    grid_max = 1.5;
    grid_min = -grid_max;
    
    z_w = 0;

    C = quaternion();
    C.x = 0.3;
    C.y = 0.5;
    C.z = 0.4;
    C.w = 0.2;

    max_iterations = 8;
    threshold = 4.0;
    
    step_size = (grid_max - grid_min) / (res - 1);

    Z = quaternion();
    Z.x = grid_min;
    Z.y = grid_min;
    Z.z = grid_min;
    Z.w = z_w;

    float_slice = torch.empty((res* res, 2*num_components), dtype=torch.float32).to(device);
    float_array = np.empty((res, res, res), dtype = np.float32);

    net = Net().to(device)

    if path.exists('weights_4_100000.pth'):
        net.load_state_dict(torch.load('weights_4_100000.pth'))
        print("loaded file successfully")
    else:
        print("Could not find file...")
        exit()

    t0 = time.perf_counter()

    for i in range(res):

        print(str(i))      
        print("init")
        init_slice(float_slice, grid_min, res, Z, step_size);
        print("done init")

        for m in range(max_iterations):

            print(m);

            float_slice.to(device)
            p = net(float_slice).cpu().detach().numpy();
            calc_slice(float_slice, grid_min, res, Z, C, step_size, p, float_array, i);

        Z.z += step_size;



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






main();

