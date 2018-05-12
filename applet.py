import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math

x = ([[x,y] for x in np.arange(-10,10,0.25) for y in np.arange(-10,10,0.25)])

frames = 90
i_vector = [math.cos(math.pi/3),math.sin(math.pi/3)]
j_vector = [-math.sin(math.pi/3),math.cos(math.pi/3)]
def sigmoid(i):
    return min(1.01/(1+math.exp(-12/frames * i + 4)),1)


fig, ax = plt.subplots()

plt.xlim(-10,10)
plt.ylim(-10,10)

scat = ax.scatter([],[],marker='.',s=0.65)
scat.set_offsets(x)

def update(i,scat):
    matrix = np.array([i_vector,j_vector]).T
    matrix = (1-sigmoid(i))*np.array([[1,0],[0,1]]) + sigmoid(i)*matrix
    ax.quiver([0,0],[0,0],[1,0],[0,1], angles='xy', scale_units='xy', scale=0.2)
    transform = np.array([matrix.dot(k) for k in x])
    scat.set_offsets(transform)
    if(i % 10 == 0):
        print("Rendering frame " + str(i) + " out of " + str(frames))
    return scat,


def init():
    return scat,

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=frames, fargs=[scat], interval=20,repeat=False, blit=True)

anim.save('the_movie7.mp4', fps=30,bitrate=5000,dpi=250)
plt.close('all')

