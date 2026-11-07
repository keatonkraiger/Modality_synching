import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import COMPLETE  # noqa: E402
data = np.load(os.path.join(COMPLETE, 'Keaton', 'OM2', 'Pressure.npy'))  # Shape: (N, 60, 21, 2)
N = data.shape[0]

# Concatenate left and right foot horizontally
frames = np.concatenate([data[:, :, :, 0], data[:, :, :, 1]], axis=2)  # (N, 60, 42)

# Create figure
fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(frames[0], cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(im, ax=ax)
ax.set_xlabel('Left Foot | Right Foot')
# Turn off xaxis ticks
ax.set_xticks([])

def update(frame):
    im.set_array(frames[frame])
    return [im]

anim = FuncAnimation(fig, update, frames=N, interval=50, blit=True)
plt.show()

# Optional: save animation
# anim.save('foot_pressure.mp4', writer='ffmpeg', fps=20)