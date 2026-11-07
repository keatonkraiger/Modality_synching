import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import COMPLETE  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# BODY25 connections
BODY_PARTS = [
    (0, 1), (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 12),
    (9, 10), (10, 11), (11, 23), (23, 22),
    (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
    # Face
    (1, 0), (0, 15), (0, 16), (15, 17), (16, 18)
]
x_y_lim = 1828.8
z_lim = 1828.8
def animate_skeleton_and_com(body25, com, com_floor, n_frames=None, threshold=0.1, interval=50):
    """
    Combined animation: BODY25 skeleton + CoM + CoM_floor.

    body25: (N, 25, 4)  -> x,y,z,conf
    com: (N, 4)         -> x,y,z,conf
    com_floor: (N, 4)   -> x,y,z,conf
    """

    N = body25.shape[0]
    n_frames = N if n_frames is None else min(n_frames, N)

    # --- Setup figure ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Fixed axis limits
    ax.set_xlim([-x_y_lim, x_y_lim])
    ax.set_ylim([-x_y_lim, x_y_lim])
    ax.set_zlim([0, z_lim])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D World: Skeleton + CoM")
    ax.view_init(elev=28, azim=49)
    ax.grid(True)

    # Draw origin + smaller axes
    L = 300  # much smaller arrows
    ax.quiver(0,0,0, L,0,0, color="r", linewidth=2)
    ax.text(L*1.2,0,0,"+X",color="r")
    ax.quiver(0,0,0, 0,L,0, color="b", linewidth=2)
    ax.text(0,L*1.2,0,"+Y",color="b")
    ax.quiver(0,0,0, 0,0,L, color="g", linewidth=2)
    ax.text(0,0,L*1.2,"+Z",color="g")

    # --- Init plot elements ---
    scat = ax.scatter([], [], [], s=30, c="red")                  # skeleton joints
    lines = [ax.plot([], [], [], c="k", linewidth=1)[0] for _ in BODY_PARTS]

    com_point, = ax.plot([], [], [], "ro", markersize=10, label="CoM", alpha=0.8)
    com_floor_point, = ax.plot([], [], [], "bo", markersize=10, label="CoM Floor", alpha=0.8)
    com_trail, = ax.plot([], [], [], "r-", linewidth=2, alpha=0.6)
    floor_trail, = ax.plot([], [], [], "b-", linewidth=2, alpha=0.6)

    ax.legend()

    def update(frame):
        # --- Skeleton ---
        joints = body25[frame, ...]  # ignore background joint
        xs, ys, zs, conf = joints.T
        mask = conf > threshold
        scat._offsets3d = (xs[mask], ys[mask], zs[mask])
        for line, (i, j) in zip(lines, BODY_PARTS):
            if conf[i] > threshold and conf[j] > threshold:
                line.set_data([xs[i], xs[j]], [ys[i], ys[j]])
                line.set_3d_properties([zs[i], zs[j]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        # --- CoM point ---
        if frame < len(com) and com[frame,3] > 0:
            x,y,z = com[frame,:3]
            com_point.set_data_3d([x],[y],[z])
        else:
            com_point.set_data_3d([],[],[])

        # --- CoM floor point ---
        if frame < len(com_floor) and com_floor[frame,3] > 0:
            x,y,z = com_floor[frame,:3]
            com_floor_point.set_data_3d([x],[y],[z])
        else:
            com_floor_point.set_data_3d([],[],[])

        # --- Trails ---
        trail_len = 100
        start = max(0, frame-trail_len)
        # CoM
        valid = (com[start:frame+1,3] > 0)
        com_trail.set_data_3d(com[start:frame+1,0][valid],
                              com[start:frame+1,1][valid],
                              com[start:frame+1,2][valid])
        # Floor
        valid = (com_floor[start:frame+1,3] > 0)
        floor_trail.set_data_3d(com_floor[start:frame+1,0][valid],
                                com_floor[start:frame+1,1][valid],
                                com_floor[start:frame+1,2][valid])

        ax.set_title(f"3D World: Skeleton + CoM  (Frame {frame}/{n_frames-1})")
        return [scat, *lines, com_point, com_floor_point, com_trail, floor_trail]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.show()
    return anim



# --- Example run for one OM dir ---
if __name__=="__main__":
    om_dir = os.path.join(COMPLETE, "Keaton", "OM2")   # just one dir for now
    body25 = np.load(os.path.join(om_dir,"BODY25_3D.npy"))
    com = np.load(os.path.join(om_dir,"CoM.npy"))
    com_floor = np.load(os.path.join(om_dir,"CoM_floor.npy"))

    animate_skeleton_and_com(body25, com, com_floor, n_frames=500)
