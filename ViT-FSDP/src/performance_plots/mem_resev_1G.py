import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

# Real
ddp = [2.9863, 14.2148, 20.2227, 61.5625]
NO_SHARD = [2.8906, 14.8008, 21.2734, 63.2109]
HYBRID = [2.8633, 14.8828, 21.8594, 63.6133]
HYBRID_2GPUs = [1.9961, 7.4902, 10.875, 30.1621]

x_axis = ['base', 'huge', '1B', '3B',]

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
axarr.set_xticks(ind)
axarr.set_xticklabels(labels,fontdict=font,)
axarr.set_ylabel('GPU Mem (GB)',fontdict=font, fontsize=14.5)
axarr.set_xlabel('Model',fontdict=font, fontsize=14.5)
plt.legend((p0[0], p1[0], p2[0], p3[0],),
           ('ddp', 'NO_SHARD', 'HYBRID_1GPU', 'HYBRID_2GPUs'),
           fontsize=15)
axarr.set_title('Max Reserved GPU Memory', fontdict=font, fontsize=14.5)
axarr.tick_params(labelsize=14.5)

#plt.show()
plt.savefig('plots/perf_mem_resev_1G.png')
