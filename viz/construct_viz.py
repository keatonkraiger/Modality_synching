import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import COMPLETE, MANUAL_CHECK  # noqa: E402
import takes  # noqa: E402
from glob import glob
import numpy as np
import subprocess

# Streams every take must have. Anything outside this list (the mocap-derived
# arrays) can legitimately be absent for a take that was never run through
# Plug-in-Gait -- see the manifest.json combine_all.py writes per take.
REQUIRED_NPY = ['BODY25_3D.npy', 'BODY25_V1.npy', 'BODY25_V2.npy', 'Pressure.npy']
OPTIONAL_NPY = ['MOCAP_3D.npy', 'MOCAP_MRK.npy', 'CoM.npy', 'CoM_floor.npy']


def available_streams(om_dir):
    """Which optional mocap streams this take actually has."""
    return {f for f in OPTIONAL_NPY if os.path.exists(os.path.join(om_dir, f))}


def check_frame_counts(om_dir):
    """Check that all data files present have matching frame counts."""
    video_files = ['Video_V1.mp4', 'Video_V2.mp4']
    
    frame_counts = {}
    
    # Required .npy files - absence really is an error
    for npy_file in REQUIRED_NPY:
        path = os.path.join(om_dir, npy_file)
        if os.path.exists(path):
            data = np.load(path)
            frame_counts[npy_file] = data.shape[0]
        else:
            raise FileNotFoundError(f"{npy_file} not found in {om_dir}")

    # Optional mocap .npy files - skipped quietly when the take does not have them
    missing_optional = []
    for npy_file in OPTIONAL_NPY:
        path = os.path.join(om_dir, npy_file)
        if os.path.exists(path):
            data = np.load(path)
            frame_counts[npy_file] = data.shape[0]
        else:
            missing_optional.append(npy_file)
    if missing_optional:
        print(f"  [!] Partial take - no {', '.join(missing_optional)}; "
              f"those visualisations will be skipped.")
    
    # Check video files using ffprobe
    for video_file in video_files:
        path = os.path.join(om_dir, video_file)
        if os.path.exists(path):
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-count_packets', '-show_entries', 'stream=nb_read_packets',
                     '-of', 'csv=p=0', path],
                    capture_output=True, text=True, check=True
                )
                frame_counts[video_file] = int(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"Warning: Could not read frame count from {video_file}: {e}")
        else:
            print(f"Warning: {video_file} not found in {om_dir}")
    
    # Print summary
    print(f"\nFrame counts for {os.path.basename(om_dir)}:")
    for file, count in frame_counts.items():
        print(f"  {file}: {count}")
    
    # Check if all match
    unique_counts = set(frame_counts.values())
    if len(unique_counts) == 1:
        print(f"✓ All files match: {unique_counts.pop()} frames")
        return True
    else:
        print(f"✗ Frame count mismatch! Found: {unique_counts}")
        return False

def visualize_pressure(pressure, save_path, fps=50, frames_to_save=None):
    """Generate pressure visualization video."""
# Concatenate left and right foot horizontally
    frames = np.concatenate([pressure[:, :, :, 0], pressure[:, :, :, 1]], axis=2)  # (N, 60, 42)
    N = frames_to_save if frames_to_save is not None else frames.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(frames[0], cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Left Foot | Right Foot')
    # Turn off xaxis ticks
    ax.set_xticks([])

    def update(frame):
        im.set_array(frames[frame])
        # Update title with frame number
        ax.set_title(f"Foot Pressure (Frame {frame}/{N-1})")
        plt.tight_layout()
        return [im]

    anim = FuncAnimation(fig, update, frames=N if frames_to_save is None else frames_to_save, interval=1000/fps, blit=True)
    anim.save(save_path, writer='ffmpeg', fps=fps)
    plt.close(fig)


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

def animate_skeleton_and_com(body25, com, com_floor, save_path, frames_to_save=None, fps=50, threshold=0.1, crop_video=True, dpi=100):
    """
    Combined animation: BODY25 skeleton + CoM + CoM_floor.

    body25: (N, 25, 4)  -> x,y,z,conf
    com: (N, 4)         -> x,y,z,conf
    com_floor: (N, 4)   -> x,y,z,conf
    """

    N = body25.shape[0]
    n_frames = N if frames_to_save is None else min(frames_to_save, N)

    # --- Setup figure ---
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
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
    scat = ax.scatter([], [], [], s=50, c="orchid", alpha=1.0, edgecolors='none')  # skeleton joints
    lines = [ax.plot([], [], [], c="black", linewidth=2, alpha=1.0)[0] for _ in BODY_PARTS]

    com_point, = ax.plot([], [], [], "o", markersize=10, label="CoM", alpha=0.8, color='forestgreen')
    com_floor_point, = ax.plot([], [], [], "o", markersize=10, label="CoM Floor", alpha=0.8, color='orange')
    com_trail, = ax.plot([], [], [], "-", linewidth=2, alpha=0.6, color='forestgreen')
    floor_trail, = ax.plot([], [], [], "-", linewidth=2, alpha=0.6, color='orange')

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

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    if crop_video:
        original_save_path = save_path
        save_path = 'tmp.mp4'
        

    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=3000)
    plt.close(fig)
    
    if crop_video:
        in_path = 'tmp.mp4' 
        crop_filter = "crop=iw*0.6:ih*0.79:iw*0.2:ih*0.09"
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-vf", crop_filter,
            "-c:a", "copy",
            original_save_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(in_path)
       
def animate_markers_and_com(markers, com, com_floor, save_path, frames_to_save=None, fps=50, threshold=0.1, crop_video=True, dpi=100):
    """
    Combined animation: BODY25 skeleton + CoM + CoM_floor.

    markers: (N, 39, 4)  -> x,y,z,conf
    com: (N, 4)         -> x,y,z,conf
    com_floor: (N, 4)   -> x,y,z,conf
    """

    N = markers.shape[0]
    n_frames = N if frames_to_save is None else min(frames_to_save, N)

    # --- Setup figure ---
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Fixed axis limits
    ax.set_xlim([-x_y_lim, x_y_lim])
    ax.set_ylim([-x_y_lim, x_y_lim])
    ax.set_zlim([0, z_lim])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D World: Markers + CoM")
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
    scat = ax.scatter([], [], [], s=30, c="slateblue", edgecolors='none', alpha=1.0)                  # skeleton joints

    com_point, = ax.plot([], [], [], "o", markersize=10, label="CoM", alpha=0.8, color='forestgreen')
    com_floor_point, = ax.plot([], [], [], "o", markersize=10, label="CoM Floor", alpha=0.8, color='orange')
    com_trail, = ax.plot([], [], [], "-", linewidth=2, alpha=0.6, color='forestgreen')
    floor_trail, = ax.plot([], [], [], "-", linewidth=2, alpha=0.6, color='orange')

    ax.legend()

    def update(frame):
        # --- Skeleton ---
        marker = markers[frame, :-1, :]  # ignore background joint
        xs, ys, zs, conf = marker.T
        mask = conf > threshold
        scat._offsets3d = (xs[mask], ys[mask], zs[mask])

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

        ax.set_title(f"3D World: Markers + CoM  (Frame {frame}/{n_frames-1})")
        return [scat, com_point, com_floor_point, com_trail, floor_trail]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    if crop_video:
        original_save_path = save_path
        save_path = 'tmp.mp4'
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=3000)
    plt.close(fig)
    
    if crop_video:
        in_path = 'tmp.mp4' 
        crop_filter = "crop=iw*0.6:ih*0.79:iw*0.2:ih*0.09"
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-vf", crop_filter,
            "-c:a", "copy",
            original_save_path
        ]
        subprocess.run(cmd, check=True)
        # remove temporary file
        os.remove(in_path)

def construct_final(video_paths, save_path, frames_to_save=None, fps=50):
    """Construct final combined visualization using ffmpeg."""
    vid_v1 = video_paths['Video_V1']
    vid_v2 = video_paths['Video_V2']
    vid_pressure = video_paths['Pressure']
    vid_skel = video_paths['Joints']
    vid_markers = video_paths['Markers']
    
    # Get dimensions of bottom row videos
    def get_video_dimensions(video_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        w, h = map(int, result.stdout.strip().split(','))
        return w, h
    
    # Get total frames
    def get_frame_count(video_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    
    total_frames = get_frame_count(vid_v1)
    if frames_to_save:
        total_frames = min(total_frames, frames_to_save)
    
    # Get dimensions
    p_w, p_h = get_video_dimensions(vid_pressure)
    s_w, s_h = get_video_dimensions(vid_skel)
    m_w, m_h = get_video_dimensions(vid_markers)
    
    # Bottom row: hstack pressure, skeleton, and markers
    bottom_width = p_w + s_w + m_w
    bottom_height = max(p_h, s_h, m_h)
    
    # Top row: scale v1 and v2 to fit side-by-side with same total width
    # Each top video gets half the bottom width
    top_single_width = bottom_width // 2
    
    # Build ffmpeg filter complex
    filter_complex = (
        # Add frame counter to top videos
        f"[0:v]scale={top_single_width}:-1:force_original_aspect_ratio=decrease,"
        f"drawtext=text='Frame\\: %{{n}}/{total_frames}':fontsize=24:fontcolor=white:"
        f"box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10[v1];"
        
        f"[1:v]scale={top_single_width}:-1:force_original_aspect_ratio=decrease,"
        f"drawtext=text='Frame\\: %{{n}}/{total_frames}':fontsize=24:fontcolor=white:"
        f"box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10[v2];"
        
        # Scale bottom row videos to same height
        f"[2:v]scale={p_w}:{bottom_height}:force_original_aspect_ratio=decrease,"
        f"pad={p_w}:{bottom_height}:(ow-iw)/2:(oh-ih)/2[pressure];"
        
        f"[3:v]scale={s_w}:{bottom_height}:force_original_aspect_ratio=decrease,"
        f"pad={s_w}:{bottom_height}:(ow-iw)/2:(oh-ih)/2[skel];"
        
        f"[4:v]scale={m_w}:{bottom_height}:force_original_aspect_ratio=decrease,"
        f"pad={m_w}:{bottom_height}:(ow-iw)/2:(oh-ih)/2[markers];"
        
        # Hstack bottom row (3 videos)
        f"[pressure][skel][markers]hstack=inputs=3[bottom];"
        
        # Hstack top row
        f"[v1][v2]hstack=inputs=2[top];"
        
        # Vstack top and bottom
        f"[top][bottom]vstack=inputs=2[out]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', vid_v1,
        '-i', vid_v2,
        '-i', vid_pressure,
        '-i', vid_skel,
        '-i', vid_markers,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23'
    ]
    
    if frames_to_save:
        cmd.extend(['-frames:v', str(frames_to_save)])
    
    cmd.append(save_path)
    
    print(f"Creating collage video: {save_path}")
    subprocess.run(cmd, check=True)
    print(f"✓ Collage saved to {save_path}")    
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=COMPLETE)
    parser.add_argument('--save_path', type=str, default=MANUAL_CHECK)
    parser.add_argument('--name', type=str, required=True, help="Subject name (e.g., 'Keaton')")
    parser.add_argument('--oms', type=str, nargs='+', default=['all'])
    parser.add_argument('--skip_step', type=str, nargs='+', default=[], choices=['pressure', 'skeleton', 'markers', 'collage'])
    parser.add_argument('--use_inst_com', action='store_true', help="Use instantaneous CoM instead of smoothed CoM.") 
    args = parser.parse_args()

    print(f"Skipping step: {args.skip_step}")
        
    selection = takes.parse_selection(args.oms)
    om_dirs = [str(d) for d in takes.take_dirs(os.path.join(args.data_path, args.name))
               if takes.selected(d.name, selection)]

    subject_save_path = os.path.join(args.save_path, args.name)
    os.makedirs(subject_save_path, exist_ok=True)
    
    for om_dir in om_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {om_dir}")
        print('='*60)
       
        om_save_path = os.path.join(subject_save_path, os.path.basename(om_dir)) 
        os.makedirs(om_save_path, exist_ok=True)
        
        # Check frame counts
        print("[INFO] Checking frame counts...")
        if check_frame_counts(om_dir):
            # Generate visualizations
            # Pressure
            pressure = np.load(os.path.join(om_dir, 'Pressure.npy'))
            pressure_save_path = os.path.join(om_save_path, f"pressure.mp4")
            if 'pressure' not in args.skip_step:
                print("[INFO] Generating pressure visualization...")
                visualize_pressure(pressure, pressure_save_path)

            # 3D skeleton + CoM -- both need a CoM, which a partial take lacks
            body25 = np.load(os.path.join(om_dir, 'BODY25_3D.npy'))
            com = com_floor = None
            if args.use_inst_com:
                inst_path = os.path.join(om_dir, 'CoM_inst.npy')
                if os.path.exists(inst_path):
                    print("[INFO] Using instantaneous CoM for visualization.")
                    com = np.load(inst_path)
                    # CoM floor is just setting z to 0 for visualization
                    com_floor = com.copy()
                    com_floor[:, 2] = 0
                else:
                    print(f"[WARN] No CoM_inst.npy in {om_dir} - run instant_com.py first.")
            else:
                com_path = os.path.join(om_dir, 'CoM.npy')
                floor_path = os.path.join(om_dir, 'CoM_floor.npy')
                if os.path.exists(com_path) and os.path.exists(floor_path):
                    com = np.load(com_path)
                    com_floor = np.load(floor_path)
                else:
                    print(f"[WARN] No CoM for this take - skipping CoM overlays.")

            skel_save_path = os.path.join(om_save_path, f"3D_joints.mp4")
            if 'skeleton' not in args.skip_step:
                if com is None:
                    print("[SKIP] 3D skeleton + CoM: no CoM for this take.")
                    skel_save_path = None
                else:
                    print("[INFO] Generating 3D skeleton + CoM visualization...")
                    animate_skeleton_and_com(body25, com, com_floor, save_path=skel_save_path)
            
            # Markers + CoM
            markers_path = os.path.join(om_dir, 'MOCAP_MRK.npy')
            if args.use_inst_com:
                markers_save_path = os.path.join(om_save_path, f"3D_markers_inst_com.mp4")
            else:
                markers_save_path = os.path.join(om_save_path, f"3D_markers.mp4")
            if 'markers' not in args.skip_step:
                if com is None or not os.path.exists(markers_path):
                    print("[SKIP] 3D markers + CoM: needs both MOCAP_MRK.npy and a CoM.")
                    markers_save_path = None
                else:
                    print("[INFO] Generating 3D markers + CoM visualization...")
                    markers = np.load(markers_path)
                    animate_markers_and_com(markers, com, com_floor, save_path=markers_save_path)

            # Collage visualization - only over the panels that were actually made
            video_paths = {
                'Video_V1': os.path.join(om_dir, 'Video_V1.mp4'),
                'Video_V2': os.path.join(om_dir, 'Video_V2.mp4'),
                'Pressure': pressure_save_path,
            }
            if skel_save_path and os.path.exists(skel_save_path):
                video_paths['Joints'] = skel_save_path
            if markers_save_path and os.path.exists(markers_save_path):
                video_paths['Markers'] = markers_save_path
            if args.use_inst_com:
                collage_path = os.path.join(om_save_path, f"final_inst_com.mp4")
            else:
                collage_path = os.path.join(om_save_path, f"final.mp4")
            if 'collage' not in args.skip_step:
                print("[INFO] Generating final collage visualization...")
                construct_final(video_paths, collage_path, fps=50)

        else:
            raise ValueError(f"Frame count mismatch in {om_dir}, skipping visualization.")