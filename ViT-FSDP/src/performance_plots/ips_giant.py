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

ddp = [89,	179,	336,	690,	1365,	2779, 5523,	10773]
NO_SHARD = [187,	373,	745,	1435,	2807,	5802, 11460,	21603]
HYBRID = [189,	383,	763,	1528,	3056,	6171, 12264, 24749]
HYBRID_2GPUs = [188,	377,	753,	1454,	2871,	5611, 9585, 20453]
full = [197,	387,	776,	1518,	2995,	5779,	7806,	4099]

# Ideal
ddp_ideal = [89*1, 89*2, 89*4, 89*8, 89*16, 89*32, 89*64, 89*128]

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
axarr.set_title('Weak Scaling of ViT-1B model on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/perf_giant.png')
