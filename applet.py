import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math

x = ([[x,y] for x in np.arange(-10,10,0.25) for y in np.arange(-10,10,0.25)])

frames = 90
i_vector = [math.cos(2*math.pi/3),math.sin(2*math.pi/3)]
j_vector = [-math.sin(4*math.pi/3),math.cos(4*math.pi/3)]
def sigmoid(i):
    return min(1.01/(1+math.exp(-10/frames * i + 4)),1)


fig, ax = plt.subplots()

plt.xlim(-10,10)
plt.ylim(-10,10)

scat = ax.scatter([],[],marker='.',s=0.65)
scat.set_offsets(x)

quiver = ax.quiver([0, 0], [0, 0], [5, 0], [0, 5],color=['g','r'], angles='xy', scale_units='xy',
                   scale=1)

def update(i,scat,quiver):
    matrix = np.array([i_vector,j_vector]).T
    matrix = (1-sigmoid(i))*np.array([[1,0],[0,1]]) + sigmoid(i)*matrix
    vector_location = np.array([5*matrix.dot([1,0]),5*matrix.dot([0,1])]).T
    quiver.set_UVC(vector_location[0],vector_location[1])
    transform = np.array([matrix.dot(k) for k in x])
    scat.set_offsets(transform)
    if(i % 10 == 0):
        print("Rendering frame " + str(i) + " out of " + str(frames))
    return scat,


def init():
    return scat,

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=frames, fargs=[scat,quiver], interval=20,repeat=False, blit=False)

anim.save('the_movie7.mp4', fps=30,bitrate=5000,dpi=250)
plt.close('all')

