import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

#ddp = []
#NO_SHARD = []
#HYBRID = []
#HYBRID_2GPUs = []
#ddp_ideal = []

# Real

#ddp = []
#NO_SHARD = []
HYBRID_2GPUs = [47,	92,	183,	352,	687,	1433]
#HYBRID_4GPUs = []
HYBRID_8GPUs = [48,	96,	189,	388,	766,	1440]
HYBRID_16GPUs = [88,	175,	349,	697,	1360]

# Ideal
#ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32]

x_axis = [1, 2, 4, 8, 16, 32]
x_axis_new = [2, 4, 8, 16, 32]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, HYBRID_2GPUs, '-o', color='black', label='HYBRID_2GPUs')
#axarr.plot(x_axis, HYBRID_4GPUs, '-o', color='g', label='HYBRID_4GPUs')
axarr.plot(x_axis, HYBRID_8GPUs, '-o', color='r', label='HYBRID_8GPUs')
axarr.plot(x_axis_new, HYBRID_16GPUs, '-o', color='b', label='HYBRID_16GPUs')
#axarr.plot(x_axis, ddp_ideal, '--o', color='black', label='ddp Ideal')

axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Weak Scaling of 4B model (5349M) on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

plt.show()
