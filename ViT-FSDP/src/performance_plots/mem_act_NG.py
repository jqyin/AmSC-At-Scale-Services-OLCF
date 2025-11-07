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
HYBRID_2GPUs = [43.5654, 0]
HYBRID_4GPUs = [23.5199, 61.7023]
HYBRID_8GPUs = [13.4388, 34.924]
HYBRID_16GPUs = [8.3733, 21.2061]

# Ideal
#ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32]

x_axis = ['5B', '15B',]

fig, axarr = plt.subplots(1,1, figsize=(6,6))
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 17,
        }

width = 0.2

#labels = ['24', '96', '384', '1536']
labels = x_axis

N = len(labels)
ind = np.arange(N)

p0 = axarr.bar(ind-width, HYBRID_2GPUs , width)
p1 = axarr.bar(ind, HYBRID_4GPUs, width)
p2 = axarr.bar(ind+width, HYBRID_8GPUs, width)
p3 = axarr.bar(ind+2*width, HYBRID_16GPUs, width)
axarr.set_xticks(ind)
axarr.set_xticklabels(labels,fontdict=font,)
axarr.set_ylabel('GPU Mem (GB)',fontdict=font, fontsize=14.5)
axarr.set_xlabel('Number of Nodes',fontdict=font, fontsize=14.5)
plt.legend((p0[0], p1[0], p2[0], p3[0],),
           ('HYBRID_2GPU', 'HYBRID_4GPUs', 'HYBRID_8GPUs', 'HYBRID_16GPUs',),
           fontsize=15, loc='upper left')
axarr.set_title('Peak Active GPU Memory', fontdict=font, fontsize=14.5)
axarr.tick_params(labelsize=14.5)

#plt.show()
plt.savefig('plots/perf_mem_act_NG.png')
