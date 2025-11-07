import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

# Real
HYBRID_2GPUs = [47,	92,	183,	352,	687,	1433, 2893, 4420]
HYBRID_4GPUs = [47,	93,	182,	362,	695,	1248,	2120, 3282]
HYBRID_8GPUs = [48,	96,	189,	388,	766,	1440, 2820, 5565]
HYBRID_16GPUs = [88,	175,	349,	697,	1360, 2726, 5277]
full = [50,	100,	197,	381,	768,	1307,	2306,	2262]
grad_op = [49,	99,	197,	389,	783,	1509,	2021,	4207]

# Ideal
HYBRID_2GPUs_ideal = [47*1, 47*2, 47*4, 47*8, 47*16, 47*32, 47*64, 47*128]

x_axis = [1, 2, 4, 8, 16, 32, 64, 128]
x_axis_new = [2, 4, 8, 16, 32, 64, 128]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, HYBRID_2GPUs, '-o', color='black', label='HYBRID_2GPUs')
axarr.plot(x_axis, HYBRID_4GPUs, '-o', color='g', label='HYBRID_4GPUs')
axarr.plot(x_axis, HYBRID_8GPUs, '-o', color='r', label='HYBRID_8GPUs')
axarr.plot(x_axis_new, HYBRID_16GPUs, '-o', color='b', label='HYBRID_16GPUs')
axarr.plot(x_axis, full, '-o', color='m', label='FULL_SHARD')
axarr.plot(x_axis, grad_op, '-o', color='y', label='SHARD_GRAD_OP')
axarr.plot(x_axis, HYBRID_2GPUs_ideal, '--o', color='black', label='HYBRID_2GPUs Ideal')

axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Weak Scaling of ViT-5B model on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/perf_5B.png')
