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

ddp = [29,	58,	119,	228,	460,	902, 1794, 3639]
NO_SHARD = [76,	153,	301,	591,	1193,	2269, 4468,	8457]
HYBRID = [79,	162,	321,	648,	1286,	2523, 5141, 10274]
HYBRID_2GPUs = [79,	155,	306,	587,	1154,	2423, 4668, 7449]
full = [83,	161,	323,	644,	1278,	2511,	3759,	3708]

# Ideal
ddp_ideal = [29*1, 29*2, 29*4, 29*8, 29*16, 29*32, 29*64, 29*128]


x_axis = [1, 2, 4, 8, 16, 32, 64, 128]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, ddp, '-o', color='black', label='ddp')
axarr.plot(x_axis, NO_SHARD, '-o', color='g', label='NO_SHARD')
axarr.plot(x_axis, HYBRID, '-o', color='r', label='HYBRID_1GPU')
axarr.plot(x_axis, HYBRID_2GPUs, '-o', color='b', label='HYBRID_2GPUs')
axarr.plot(x_axis, full, '-o', color='m', label='FULL_SHARD')
axarr.plot(x_axis, ddp_ideal, '--o', color='black', label='ddp Ideal')

axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Weak Scaling of ViT-3B model on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/perf_enormous.png')
