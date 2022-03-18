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






@numba.njit
def calc_muls(float_slice, res):

    p = np.zeros((res* res, num_components), dtype=np.float64);

    for j in range(res):

       for k in range(res):

           index = j*res + k
           p[index][0] = float_slice[index][0]*float_slice[index][0] - float_slice[index][1]*float_slice[index][1] - float_slice[index][2]*float_slice[index][2] - float_slice[index][3]*float_slice[index][3];
           p[index][1] = 2 * float_slice[index][0]*float_slice[index][1];
           p[index][2] = 2 * float_slice[index][0]*float_slice[index][2];
           p[index][3] = 2 * float_slice[index][0]*float_slice[index][3];

           
    return p;




@numba.njit
def calc_slice(float_slice, grid_min, res, Z_x, Z_y, Z_z, Z_w, C_x, C_y, C_z, C_w, step_size, p, float_array, i):

    Z_x = grid_min;

    for j in range(res):

       Z_y = grid_min;

       for k in range(res):

           index = j*res + k

           p[index][0] += C_x;
           p[index][1] += C_y;
           p[index][2] += C_z;
           p[index][3] += C_w;

           float_array[i][j][k] = math.sqrt(p[index][0]*p[index][0] + p[index][1]*p[index][1] + p[index][2]*p[index][2] + p[index][3]*p[index][3]);
   
           float_slice[index][0] = p[index][0];
           float_slice[index][1] = p[index][1];
           float_slice[index][2] = p[index][2];
           float_slice[index][3] = p[index][3];
           float_slice[index][4] = p[index][0];
           float_slice[index][5] = p[index][1];
           float_slice[index][6] = p[index][2];
           float_slice[index][7] = p[index][3];
  
           Z_y += step_size;

       Z_x += step_size;



@numba.njit
def init_slice(float_slice, grid_min, res, Z_x, Z_y, Z_z, Z_w, step_size):

    Z_x = grid_min;

    for j in range(res):

        Z_y = grid_min;

        for k in range(res):
        
            index = j*res + k;

            float_slice[index][0] = Z_x;
            float_slice[index][1] = Z_y;
            float_slice[index][2] = Z_z;
            float_slice[index][3] = Z_w;
            float_slice[index][4] = Z_x;
            float_slice[index][5] = Z_y;
            float_slice[index][6] = Z_z;
            float_slice[index][7] = Z_w;

            Z_y += step_size;

        Z_x += step_size;


def main():

    res = 50;
    grid_max = 1.5;
    grid_min = -grid_max;
    
    z_w = 0;

    #C = quaternion();
    C_x = 0.3;
    C_y = 0.5;
    C_z = 0.4;
    C_w = 0.2;

    max_iterations = 8;
    threshold = 4.0;
    
    step_size = (grid_max - grid_min) / (res - 1);

    #Z = quaternion();
    Z_x = grid_min;
    Z_y = grid_min;
    Z_z = grid_min;
    Z_w = z_w;

    float_slice = np.empty((res* res, 2*num_components), dtype=np.float64);
    float_array = np.empty((res, res, res), dtype = np.float32);

    t0 = time.perf_counter()

    for i in range(res):

        print(str(i))      
        print("init")
        init_slice(float_slice, grid_min, res, Z_x, Z_y, Z_z, Z_w, step_size);
        print("done init")

        for m in range(max_iterations):

            print(m);

            p = calc_muls(float_slice, res)# net(torch.from_numpy(float_slice)).detach().numpy();
            calc_slice(float_slice, grid_min, res, Z_x, Z_y, Z_z, Z_w, C_x, C_y, C_z, C_w, step_size, p, float_array, i);

        Z_z += step_size;



    t1 = time.perf_counter()

    print("Time elapsed: ", t1 - t0)



    verts, faces, normals, values = measure.marching_cubes(float_array, threshold, spacing=(step_size, step_size, step_size))

    thefile = open('test_cpu_fast.obj', 'w')
    for item in verts:
      thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
      thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in faces:
      thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0] + 1, item[1] + 1, item[2] + 1))  

    thefile.close()






main();

