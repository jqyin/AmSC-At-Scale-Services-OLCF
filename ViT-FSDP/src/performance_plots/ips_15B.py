import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

# Real
HYBRID_4GPUs = [15,	30,	60,	120,	238,	473,	928,	1649]
HYBRID_8GPUs = [16,	32,	63,	127,	245,	502,	982,	1878]
HYBRID_16GPUs = [31,	62,	125,	250,	500,	984,	1789]
full = [18,	36,	71,	138,	283,	560,	1113,	1754]
grad_op = [18,	37,	73,	147,	271,	575,	1145,	2196]

# Ideal
HYBRID_4GPUs_ideal = [15*1, 15*2, 15*4, 15*8, 15*16, 15*32, 15*64, 15*128]

x_axis = [1, 2, 4, 8, 16, 32, 64, 128]
x_axis_new = [2, 4, 8, 16, 32, 64, 128]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, HYBRID_4GPUs, '-o', color='g', label='HYBRID_4GPUs')
axarr.plot(x_axis, HYBRID_8GPUs, '-o', color='r', label='HYBRID_8GPUs')
axarr.plot(x_axis_new, HYBRID_16GPUs, '-o', color='b', label='HYBRID_16GPUs')
axarr.plot(x_axis, full, '-o', color='m', label='FULL_SHARD')
axarr.plot(x_axis, grad_op, '-o', color='y', label='SHARD_GRAD_OP')
axarr.plot(x_axis, HYBRID_4GPUs_ideal, '--o', color='black', label='HYBRID_4GPUs Ideal')

axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Weak Scaling of ViT-15B model on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/perf_15B.png')
