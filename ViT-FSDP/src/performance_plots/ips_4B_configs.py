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
Non = [347, 280, 337]
BACKWARD_POST = [369, 315, 347]
BACKWARD_PRE = [377, 288, 348]
Non_1 = [351, 335, 335]
BACKWARD_POST_1 = [375, 361, 347]
BACKWARD_PRE_1 = [388, 360, 352]

x_axis = ['FULL', 'HYBRID_2GPUs', 'HYBRID_16GPUs']

fig, axarr = plt.subplots(1,1, figsize=(9,5))
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 15,
        }

width = 0.12

labels = x_axis

N = len(labels)
ind = np.arange(N)

p0 = axarr.bar(ind-width, Non , width)
p1 = axarr.bar(ind, BACKWARD_PRE , width)
p2 = axarr.bar(ind+width, BACKWARD_POST , width)
p3 = axarr.bar(ind+2*width, Non_1 , width)
p4 = axarr.bar(ind+3*width, BACKWARD_PRE_1 , width)
p5 = axarr.bar(ind+4*width, BACKWARD_POST_1 , width)
axarr.set_xticks(ind)
axarr.set_xticklabels(labels,fontdict=font,)
axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font, fontsize=14.5)
axarr.set_xlabel('Sharding Strategy',fontdict=font, fontsize=14.5)
axarr.legend((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0]),
             ('None', 'BACKWARD_PRE', 'BACKWARD_POST',
              'None+limit_all_gathers', 'BACKWARD_PRE+limit_all_gathers', 'BACKWARD_POST+limit_all_gathers',),
             fontsize=13, loc='lower right')
axarr.set_title('Sharding Strategy & Prefetching Tests on Synthetic', fontdict=font, fontsize=14.5)
axarr.tick_params(labelsize=14.5)

#plt.show()
plt.savefig('plots/perf_configs.png')
