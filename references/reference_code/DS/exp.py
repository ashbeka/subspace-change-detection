from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from config import Confing
from utils import read_c3d,gen_shape_subspace,cal_magnitude,gen_shape_difference_subspace


path = "../dataset/07_01.c3d"
cfg = Confing()
tau = cfg.interval
data = read_c3d(path)
num_frame = data.shape[2]


mag_list = []
frame_list = []
f = tau*2 // 2

for i in range(num_frame-tau*2):

    S1 = gen_shape_subspace(data[:,:,i],cfg)
    S2 = gen_shape_subspace(data[:,:,i+tau],cfg)
    S3 = gen_shape_subspace(data[:,:,i+tau*2],cfg)

    D1 = gen_shape_difference_subspace(S1,S2,cfg)
    D2 = gen_shape_difference_subspace(S2,S3,cfg)


    mag = cal_magnitude(D1,D2)
    mag_list.append(mag)
    frame_list.append(f)
    f += 1


plt.figure(figsize=(10,4))
plt.plot(frame_list, mag_list)
plt.xlabel('frame')
plt.ylabel('mag')
plt.grid(True)
plt.show()
#plt.savefig(save_path)