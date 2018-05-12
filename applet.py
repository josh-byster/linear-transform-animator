import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math
x = ([[x,y] for x in np.arange(-10,10,0.25) for y in np.arange(-10,10,0.25)])

frames = 300

def sigmoid(i):
    return 1/(1+math.exp(-8/frames * i + 4))

print(sigmoid(1))
transform = np.array([matrix.dot(k) for k in x])

fig, ax = plt.subplots()

plt.xlim(-10,10)
plt.ylim(-10,10)

transform_x, transform_y = transform.T
scat = ax.scatter(transform_x,transform_y,marker='.',s=0.7)

def update(i,scat):
    matrix = np.array([[3,1],[3,1]])
    matrix = (1-sigmoid(i))*np.array([[1,0],[0,1]]) + sigmoid(i)*matrix
    transform = np.array([matrix.dot(k) for k in x])
    scat.set_offsets(transform)
    if(i % 50 == 0):
        print("Rendering frame " + str(i) + " out of " + str(frames))
    return scat,


def init():
    return scat,

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=frames, fargs=[scat], interval=20,repeat=False, blit=True)

anim.save('the_movie.mp4', fps=50,bitrate=5000,dpi=250)
plt.close('all')

