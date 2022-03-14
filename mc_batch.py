import numpy as np
import math
import torch
import os.path

from numpy import random
from skimage import measure
from torch.autograd import Variable
from os import path



num_components = 4 # quaternions

class quaternion:
    x = 0;
    y = 0;
    z = 0;
    w = 0;

    def __str__(self): 
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.w);

def quat_mul(left, right):
    ret = quaternion();

    ret.x = left.x*right.x - left.y*right.y - left.z*right.z - left.w*right.w;
    ret.y = left.x*right.y + left.y*right.x + left.z*right.w - left.w*right.z;
    ret.z = left.x*right.z - left.y*right.w + left.z*right.x + left.w*right.y;
    ret.w = left.x*right.w + left.y*right.z - left.z*right.y + left.w*right.x;

    return ret;


def get_predictions()

def quat_mul_ai(left, right):

    ret = quaternion();

    batch = torch.zeros(num_components*2)
    batch[0] = left.x
    batch[1] = left.y
    batch[2] = left.z
    batch[3] = left.w

    batch[4] = right.x
    batch[5] = right.y
    batch[6] = right.z
    batch[7] = right.w

#    print(batch)

    prediction = net(batch).detach().numpy()

    ret.x = prediction[0];
    ret.y = prediction[1];
    ret.z = prediction[2];
    ret.w = prediction[3];


#    print(prediction)

    return ret;


def quat_add(left, right):
    ret = quaternion();

    ret.x = left.x + right.x;
    ret.y = left.y + right.y;
    ret.z = left.z + right.z;
    ret.w = left.w + right.w;

    return ret;



def self_dot(q):
    return q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    
def magnitude(q):
    return math.sqrt(self_dot(q));

def iterate(Z):
    for i in range(max_iterations):

        Z = quat_mul(Z, Z);
        Z = quat_add(Z, C);
        
        if magnitude(Z) >= threshold:
            break;
    
    return magnitude(Z);



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

if path.exists('weights_4_100000.pth'):
    net.load_state_dict(torch.load('weights_4_100000.pth'))
    print("loaded file successfully")
else:
    print("Could not find file...")
    exit()
          




res = 50;
float_array = np.zeros((res, res, res), np.float32)

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

for i in range(z_res):

    print(str(i))

    float_slice_a = np.empty((res, res), dtype=quaternion)
    float_slice_b = np.empty((res, res), dtype=quaternion)
    float_slice_answer = np.empty((res, res), dtype=quaternion)
    float_slice_magnitude = np.empty((res, res), dtype=quaternion)

    Z.x = x_grid_min;
    
    for j in range(x_res):
        Z.y = y_grid_min;

        for k in range(y_res):
        
            float_slice_a[j][k] = Z;
            float_slice_b[j][k] = Z;

            Z.y += y_step_size;

        Z.x += x_step_size;

    for m in range(max_iterations):

        Z.x = x_grid_min;

        for j in range(x_res):
            Z.y = y_grid_min;

            for k in range(y_res):
                float_slice_answer[j][k] = quat_add(quat_mul_ai(float_slice_a[j][k], float_slice_b[j][k]), C);
                float_slice_magnitude[j][k] = magnitude(float_slice_answer[j][k]);
                float_slice_a[j][k] = float_slice_b[j][k] = float_slice_answer[j][k];

                Z.y += y_step_size;

            Z.x += x_step_size;

    Z.z += z_step_size;

    float_array[i] = float_slice_magnitude







verts, faces, normals, values = measure.marching_cubes(float_array, threshold)

thefile = open('test.obj', 'w')
for item in verts:
  thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

for item in normals:
  thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

for item in faces:
  thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0] + 1, item[1] + 1, item[2] + 1))  

thefile.close()



