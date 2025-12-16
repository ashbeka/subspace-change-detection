from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import read_c3d, display_motion,display_point


path = "../dataset/16_57.c3d"

data = read_c3d(path)
display_motion(path)
#display_point(path,0)
