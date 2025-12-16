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



#収束のエラーが出た場合の特異値分解計算
def error_cal_svd(A):
    
    w,v = scipy.linalg.eigh(np.dot(A.T ,A))
    
    w = w[::-1]
    w[w < 0] = 0
    v = v[:,::-1]
    
    Y = np.sqrt(w) #Y:特異値
    X = np.dot(A,v) #X:左特異ベクトル行列

    epsilon = 0  # ゼロとみなす小さな値の閾値
    Y_inv = np.where(Y > epsilon, 1.0 / Y, 0)
    
    X = np.dot(X,np.diag(Y_inv)) #
    Z = v.T #Z:右特異ベクトルの転置行列

    return X, Y, Z


#収束エラーを考慮した特異値分解
def cal_svd(A):
    try:
        U, S, V = np.linalg.svd(A)
        return U,S,V
    except:
        U, S, V  = error_cal_svd(A)
        return U,S,V


#1frame内のポイントデータから形状部分空間を作成する。
def gen_shape_subspace(data, cfg):
    #data shape is (3, num)
    X = data.T
    
    mv = np.mean(X, axis=0)
    Xc = X - mv
    U, S, V = cal_svd(Xc)

    return U[:,0:cfg.subspace_dim]

#差分部分空間の生成
def gen_shape_difference_subspace(S1,S2,cfg):
    # U, S, Vt = cal_svd(S1.T @ S2)
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
    _, S, _ = cal_svd(S1.T @ S2)
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


    U,S,V = np.linalg.svd(S1)
    print(S)
    U,S,V = error_cal_svd(S1)
    print(S)
    U,S,V = cal_svd(S1)
    print(S)





