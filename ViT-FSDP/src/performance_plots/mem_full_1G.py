import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

# Real
base = [0.9464,	0.8839,	0.8464,	0.8348,	0.8263,	0.8221,	0.82,	0.819]
huge = [2.6464,	2.1822,	1.9529,	1.8382,	1.7763,	1.7474,	1.7329,	1.7257]
giant = [3.4272,	2.748,	2.4185,	2.2505,	2.1661,	2.1245,	2.1037,	2.0934]
enormous = [8.9501,	6.2753,	5.1783,	4.598,	4.3203,	4.184,	4.1106,	4.0757]

# Ideal
#ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32, 853*64, 853*128]

x_axis = [1, 2, 4, 8, 16, 32, 64, 128]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis, base, '-o', color='black', label='base')
axarr.plot(x_axis, huge, '-o', color='g', label='huge')
axarr.plot(x_axis, giant, '-o', color='r', label='1B')
axarr.plot(x_axis, enormous, '-o', color='b', label='3B')

axarr.set_ylabel('GPU Mem (GB)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Peak Active GPU Memory (FULL_SHARD)', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/mem_full_1G.png')
