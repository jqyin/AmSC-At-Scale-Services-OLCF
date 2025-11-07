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

ddp = [120,	239,	478,	934,	1844,	3740]
NO_SHARD = [239,	472,	948,	1843,	3669,	7430]
HYBRID = [244,	485,	975,	1942,	3877,	7767]
HYBRID_2GPUs = [236,	476,	949,	1890,	3624,	7412]

# Ideal
ddp_ideal = [120*1, 120*2, 120*4, 120*8, 120*16, 120*32]

x_axis = [1, 2, 4, 8, 16, 32]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, ddp, '-o', color='black', label='ddp')
axarr.plot(x_axis, NO_SHARD, '-o', color='g', label='NO_SHARD')
axarr.plot(x_axis, HYBRID, '-o', color='r', label='HYBRID_1GPU')
axarr.plot(x_axis, HYBRID_2GPUs, '-o', color='b', label='HYBRID_2GPUs')
axarr.plot(x_axis, ddp_ideal, '--o', color='black', label='ddp Ideal')

axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Weak Scaling of Huge model (635M) on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

plt.show()
