from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import eig
from config import Confing
import scipy
from matplotlib import colors


def read_c3d(path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    data = point_data[0:3,:,:] #(XYZ, num_mark, num_frame)
    return data

def display_point(path, frame):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    data = point_data[0:3,:,frame]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ax.set_box_aspect([1, 1, 2])
    plt.show()

def display_motion(path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax.set_title(f"frame num {0}")

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        ft.set_text(f"frame num {frame}")
        return sc

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()


def display_motion_score(path, x, y, save_path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    title = path.split('/')[2]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_xlabel('value')
    ax2.set_title(title)
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]],y[0:frame-x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    #plt.show()
    ani.save(save_path, writer='pillow', fps=20)


def display_motion_score_contribution(path, x, y, contribution, save_path):

    title = path.split('/')[2]

    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]
    cb = contribution[0]
    norm = colors.Normalize(vmin=min(cb), vmax=max(cb))

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], c=cb, norm=norm, cmap='jet', s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_xlabel('value')
    ax2.set_title(title)
    ax2.grid(True)

    
    cbar = plt.colorbar(sc, pad=0.2)
    cbar.set_ticks([])


    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])

        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]+1],y[0:frame-x[0]+1])
            cb = contribution[frame-x[0]+1]
            norm = colors.Normalize(vmin=min(cb), vmax=max(cb))
            sc.set_array(cb)
            sc.set_norm(norm)
        ft.set_text(f"frame num {frame}")
        cbar.set_ticks([])
        
        return sc, line, cbar

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    #plt.show()
    ani.save(save_path, writer='pillow', fps=20)



#複数の波形を表示する
def display_motion_some_score(path, x, y, y_label, save_path):

    # yはarray
    #y[0] 値1
    #y[1] 値2
    #y_label[0] 値1のラベル
    #y_label[1] 値2のラベル

    title = path.split('/')[2]

    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line = []
    for i in range(len(y)):
        line.append(ax2.plot([], [], label=y_label[i]))

    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_xlabel('value')
    ax2.set_title(title)
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])

        if x[0]<=frame and frame < x[-1]:
            for i in range(len(y)):
                line[i][0].set_data(x[0:frame-x[0]+1],y[i][0:frame-x[0]+1])
        ft.set_text(f"frame num {frame}")
        
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)
    plt.legend()
    #plt.show()
    ani.save(save_path, writer='pillow', fps=20)



#1frame内のポイントデータから形状部分空間を作成する。
def gen_shape_subspace(data, cfg):
    #data shape is (3, num)
    X = data.T
    mv = np.mean(X, axis=0)
    Xc = X - mv
    U, S, V = np.linalg.svd(Xc)

    return U[:,0:cfg.subspace_dim]

#差分部分空間の生成
def gen_shape_difference_subspace(S1,S2,cfg):
    # U, S, Vt = np.linalg.svd(S1.T @ S2)
    # S = np.diag(S)
    # I = np.eye(S.shape[0],S.shape[1])
    # D = ((S1 @ U) - (S2 @ Vt.T)) @ ((2 * (I - S))**(-0.5))
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eig(G)
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    return eigen_vec[:,idx]

#共通部分空間の作成
def gen_shape_principal_com_subspace(S1,S2,cfg):
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eig(G)
    idx = np.where(1 <= eigen_val)[0]
    return eigen_vec[:,idx]

#部分空間同士のマグニチュード(類似度)を計算
def cal_magnitude(S1,S2):
    _, S, _ = np.linalg.svd(S1.T @ S2)
    mag = np.sum(2*(1 - S))
    return mag


def gram_schmidt(arr):
    arr = np.array(arr, dtype=np.float64)
    k = arr.shape[1]
    u = arr[:,[0]]
    q = u / scipy.linalg.norm(u)

    for j in range(1, k):
        u = arr[:,[j]]
        for i in range(j):
            u -= np.dot(q[:,i], arr[:,j]) * q[:,[i]]
        qi = u / scipy.linalg.norm(u)
        q = np.append(q, qi, axis=1)
    return q

#D(M, S2')の大きさが２階差分部分空間の測地線に沿った変動成分１
def along_geodesic(S1,S2,S3,cfg):
    #６次元の和空間W(S1,S3)を作成
    W = np.concatenate([S1, S3], 1)
    W = W / np.linalg.norm(W, axis=0)
    W = gram_schmidt(W)

    #部分空間S2の３本の基底を６次元の和空間W(S1,S3)に射影する
    P = W @ W.T
    V = P @ S2

    #射影された3本の基底に対してグラムシュミット直交化を適用してV:(S2’)を求める
    #射影した基底ベクトルを正規化
    V = V / np.linalg.norm(V, axis=0)
    #グラムシュミット直交化
    V = gram_schmidt(V)

    #Mをもとめる
    M = gen_shape_principal_com_subspace(S1,S3,cfg)

    #変動成分：D(M, S2')の大きさが２階差分部分空間の測地線に沿った変動成分１
    mag = cal_magnitude(M,V)

    return mag


#D(S2, S2')の大きさが２階差分部分空間の測地線に直交する変動成分２
def orth_decomposition_geodesic(S1,S2,S3,cfg):
    #６次元の和空間W(S1,S3)を作成
    W = np.concatenate([S1, S3], 1)
    W = W / np.linalg.norm(W, axis=0)
    W = gram_schmidt(W)

    #部分空間S2の３本の基底を６次元の和空間W(S1,S3)に射影する
    P = W @ W.T
    V = P @ S2

    #射影された3本の基底に対してグラムシュミット直交化を適用してV:(S2’)を求める
    #射影した基底ベクトルを正規化
    V = V / np.linalg.norm(V, axis=0)
    #グラムシュミット直交化
    V = gram_schmidt(V)

    #変動成分：D(S2, S2')の大きさが２階差分部分空間の測地線に直交する変動成分
    mag = cal_magnitude(S2,V)

    return mag




if __name__ == "__main__": 
    S1 = np.array([[1,2,3],[1,2,3],[1,2,3]])
    S2 = np.array([[4,5,6],[4,5,6],[4,5,6]])
    print(S1)
    print(S2)

    W = np.concatenate([S1, S2], 1)
    print(W)





