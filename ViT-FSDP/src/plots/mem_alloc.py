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
ddp = [2.4721, 13.2812, 18.7437, 59.8159, 0]
NO_SHARD = [1.9594, 10.4048, 14.691, 47.3326, 0]
HYBRID = [1.958, 10.4049, 14.6917, 47.3302, 0]
HYBRID_2GPUs = [1.3225, 5.6956, 7.9241, 24.6311, 41.8989]
HYBRID_8GPUs = [0, 0, 0, 0, 12.2433]
HYBRID_16GPUs = [0, 0, 0, 0, 7.5339]

# Ideal
#ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32]

x_axis = ['base', 'huge', 'giant', 'enormous', '4B']

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

p0 = axarr.bar(ind-width, ddp , width)
p1 = axarr.bar(ind, NO_SHARD , width)
p2 = axarr.bar(ind+width, HYBRID , width)
p3 = axarr.bar(ind+2*width, HYBRID_2GPUs , width)
p4 = axarr.bar(ind+3*width, HYBRID_8GPUs , width)
p5 = axarr.bar(ind+4*width, HYBRID_16GPUs , width)
axarr.set_xticks(ind)
axarr.set_xticklabels(labels,fontdict=font,)
axarr.set_ylabel('GPU Mem (GB)',fontdict=font, fontsize=14.5)
axarr.set_xlabel('Number of Nodes',fontdict=font, fontsize=14.5)
plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0]),
           ('ddp', 'NO_SHARD', 'HYBRID_1GPU', 'HYBRID_2GPUs', 'HYBRID_8GPUs', 'HYBRID_16GPUs',),
           fontsize=15)
axarr.set_title('Max Allocate GPU Memory', fontdict=font, fontsize=14.5)
axarr.tick_params(labelsize=14.5)

plt.show()

