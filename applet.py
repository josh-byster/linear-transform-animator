import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math
x = ([[x,y] for x in range(-10,10) for y in range (-10,10)])

frames = 300

def sigmoid(i):
    return 1/(1+math.exp(-8/frames * i + 4))

print(sigmoid(1))
transform = np.array([matrix.dot(k) for k in x])

fig, ax = plt.subplots()

plt.xlim(-10,10)
plt.ylim(-10,10)

transform_x, transform_y = transform.T
scat = ax.scatter(transform_x,transform_y,marker='+')

def update(i,scat):
    matrix = np.array([[1,1],[0,1]])
    matrix = (1-sigmoid(i))*np.array([[-2,0],[0,-1]]) + sigmoid(i)*matrix
    transform = np.array([matrix.dot(k) for k in x])
    scat.set_offsets(transform)
    return scat,


def init():
    return scat,

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=300, fargs=[scat], interval=20,repeat=False, blit=True)

anim.save('the_movie.mp4', fps=50,bitrate=10000,dpi=500)
plt.close('all')

