import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math

# Creates points for a grid from -10 to 10 spaced 0.25 units apart
x = ([[x, y] for x in np.arange(-10, 10, 0.25) for y in np.arange(-10, 10, 0.25)])

# Number of frames in the animation
frames = 150

# Where the i unit-vector lands
i_vector = [3, -2]

# Where the j unit-vector lands
j_vector = [2, 1]


# Smoothing function that acts similar to Bezier smoothing with keyframes.
# Returns how complete the transformation is on a scale from 0 to 1,
# based on the current and total frame count
def sigmoid(i):
    return min(1.01 / (1 + math.exp(-10 / frames * i + 4)), 1)


fig, ax = plt.subplots()

plt.xlim(-10, 10)
plt.ylim(-10, 10)

scat = ax.scatter([], [], marker='.', s=0.65)
scat.set_offsets(x)

# Plots the two unit vectors to show what happens to the scaling
quiver = ax.quiver([0, 0], [0, 0], [1, 0], [0, 1], color=['g', 'r'], angles='xy', scale_units='xy',
                   scale=0.2)


def update(i):
    matrix = np.array([i_vector, j_vector]).T

    # Compute a matrix that is in the middle between the full transformation matrix and the identity
    matrix = (1 - sigmoid(i)) * np.array([[1, 0], [0, 1]]) + sigmoid(i) * matrix

    # Set vector location - must transpose since we need U and V representing x and y components
    # of each vector respectively (without transposing, each column represents each unit vector)
    vector_location = np.array([matrix.dot([1, 0]), matrix.dot([0, 1])]).T
    quiver.set_UVC(vector_location[0], vector_location[1])
    transform = np.array([matrix.dot(k) for k in x])
    scat.set_offsets(transform)
    if i % 10 == 0:
        print("Rendering frame " + str(i) + " out of " + str(frames))


anim = animation.FuncAnimation(fig, update, frames=frames, interval=20, repeat=False, blit=False)

anim.save('example.mp4', fps=30, bitrate=5000, dpi=250)
