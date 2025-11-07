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

ddp = [853,	1686,	3310,	6590,	13006,	26344]
NO_SHARD = [1207,	2333,	4803,	9069,	17824,	34750]
HYBRID = [1203,	2476,	4969,	9927,	19835,	39738]
HYBRID_2GPUs = [1152,	2282,	4427,	9059,	17512,	34024]

# Ideal
ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32]

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
axarr.set_title('Weak Scaling of Base model (87M) on Synthetic Data', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

plt.show()
