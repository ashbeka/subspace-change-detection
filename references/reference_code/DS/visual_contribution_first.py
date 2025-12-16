import sys
from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from config import Confing
from utils import read_c3d,gen_shape_subspace,cal_magnitude,gen_shape_difference_subspace, display_motion_score,gram_schmidt,display_motion_score_contribution
import numpy as np
from tqdm import tqdm

# コマンドライン引数からパスを取得
if len(sys.argv) < 2:
    print("Usage: python xxx.py [data_path]")
    sys.exit(1)

path = sys.argv[1]
#path = "../dataset/07_01.c3d"

cfg = Confing()
tau = cfg.interval
data = read_c3d(path)
num_frame = data.shape[2]

data_title = path.split('/')[2].split('.')[0]

mag_list = []
frame_list = []
f = tau // 2


contribution_list = []

for i in tqdm(range(num_frame-tau*2)):

    S1 = gen_shape_subspace(data[:,:,i],cfg)
    S2 = gen_shape_subspace(data[:,:,i+tau],cfg)

    #calc difference subspace
    D = gen_shape_difference_subspace(S1,S2,cfg)

    #各部分空間の基底ベクトルをDSに射影する。
    P = D @ D.T
    V = P @ S1

    #射影した基底ベクトルを正規化
    V = V / np.linalg.norm(V, axis=0)

    #グラムシュミット直交化
    V = gram_schmidt(V)
    
    #Vの各要素を2乗
    V = np.square(V)
    V = np.sum(V, axis=1)

    mag = cal_magnitude(S1,S2)
    mag_list.append(mag)
    frame_list.append(f)
    f += 1

    contribution_list.append(V)

display_motion_score_contribution(path,frame_list,mag_list,contribution_list, f"../result/{data_title}_first.gif")



