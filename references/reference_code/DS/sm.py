from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from config import Confing
from utils import read_c3d,gen_shape_subspace,cal_magnitude,gen_shape_difference_subspace, display_motion_score


path = "../dataset/16_17.c3d"
cfg = Confing()
tau = cfg.interval
data = read_c3d(path)
num_frame = data.shape[2]


mag_list = []
frame_list = []
f = tau // 2

for i in range(num_frame-tau*2):

    S1 = gen_shape_subspace(data[:,:,i],cfg)
    S2 = gen_shape_subspace(data[:,:,i+tau],cfg)

    mag = cal_magnitude(S1,S2)
    mag_list.append(mag)
    frame_list.append(f)
    f += 1

display_motion_score(path,frame_list,mag_list, "../result/test.gif")
# plt.figure(figsize=(10,4))
# plt.plot(frame_list, mag_list)
# plt.xlabel('frame')
# plt.ylabel('mag')
# plt.grid(True)
# plt.show()
#plt.savefig(save_path)

