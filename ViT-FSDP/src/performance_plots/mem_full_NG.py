import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

# Real
A5B_f = [13.4388,	8.3723,	6.4022,	5.4088,	4.9325,		4.572,	4.5122]
A15B_f = [34.924,	21.2061,	14.9059,	12.2638,	10.9442,	10.2811,	9.9517	, 9.7874]
A5B_g = [22.3427,	18.061,	15.9152,	14.806,	14.2689,	14.0914,	13.8636,	14.6251]
A15B_g = [57.114,	46.871,	40.8138,	37.8149,	36.3178,	35.5677, 35.5649, 35.786]

# Ideal
#ddp_ideal = [853*1, 853*2, 853*4, 853*8, 853*16, 853*32, 853*64, 853*128]

x_axis_1 = [1, 2, 4, 8, 16, 64, 128]
x_axis = [1, 2, 4, 8, 16, 32, 64, 128]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(x_axis_1, A5B_f, '-o', color='black', label='5B/FULL_SHARD')
axarr.plot(x_axis, A5B_g, '-o', color='r', label='5B/SHARD_GRAD_OP')
axarr.plot(x_axis, A15B_f, '-o', color='g', label='15B/FULL_SHARD')
axarr.plot(x_axis, A15B_g, '-o', color='b', label='15B/SHARD_GRAD_OP')

axarr.set_ylabel('GPU Mem (GB)',fontdict=font)
axarr.set_xlabel('Number of Nodes',fontdict=font)
axarr.set_title('Peak Active GPU Memory', fontdict=font)
axarr.legend()
axarr.grid()
axarr.legend(prop={'size': 12})

#plt.show()
plt.savefig('plots/mem_full_NG.png')
