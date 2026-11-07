# get_model_output.py takes the output json from Vicon and extracts mocap, com, etc.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from glob import glob
import json   
import os
import pandas as pd
from collections import OrderedDict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import RAW, WORK  # noqa: E402

def create_com_animation(com_data, com_floor_data, title="CoM Animation", save_path=None, N_frames=None):
    """
    Create 3D animation of Center of Mass data
    
    Parameters:
    com_data: numpy array of shape (N_frames, 4) with [X, Y, Z, confidence]
    com_floor_data: numpy array of shape (N_frames, 4) with [X, Y, Z, confidence] 
    title: string title for the animation
    save_path: optional path to save animation as MP4
    N_frames: int or None - number of frames to render, if None uses all available frames
    """
    
    # Determine frame range
    max_frames = max(len(com_data) if com_data is not None else 0, 
                     len(com_floor_data) if com_floor_data is not None else 0)
    
    if max_frames == 0:
        print("No data found!")
        return
    
    # Use N_frames if specified, otherwise use all frames
    frames_to_render = N_frames if N_frames is not None else max_frames
    frames_to_render = min(frames_to_render, max_frames)  # Don't exceed available data
    
    print(f"Animating {frames_to_render} frames out of {max_frames} available")
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw origin + axes
    ax.scatter([0],[0],[0], c='k', marker='x', s=200, label='Origin')
    L = 1500
    ax.quiver(0,0,0, L,0,0, color='r', arrow_length_ratio=0.1, linewidth=3)
    ax.text(L*1.1,0,0,'+X',color='r')
    ax.quiver(0,0,0, 0,L,0, color='b', arrow_length_ratio=0.1, linewidth=3)
    ax.text(0,L*1.1,0,'+Y',color='b')
    ax.quiver(0,0,0, 0,0,L, color='g', arrow_length_ratio=0.1, linewidth=3)
    ax.text(0,0,L*1.1,'+Z',color='g')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D World - ' + title)
    
    max_range = 6000
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])
    ax.grid(True)
    ax.view_init(elev=28, azim=88)
    
    # Initialize empty line objects for animation
    com_point, = ax.plot([], [], [], 'ro', markersize=10, label='CoM', alpha=0.8)
    com_floor_point, = ax.plot([], [], [], 'bo', markersize=10, label='CoM Floor', alpha=0.8)
    com_trail, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.6, label='CoM Trail')
    floor_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Floor Trail')
    
    ax.legend()
    
    # Animation function
    def animate(frame):
        # Update CoM point
        if com_data is not None and frame < len(com_data):
            if com_data[frame, 3] > 0:
                x, y, z = com_data[frame, :3]
                com_point.set_data_3d([x], [y], [z])
                com_point.set_color('red')
                com_point.set_markersize(10)
            else:
                # Plot grey dot at origin when no valid CoM
                com_point.set_data_3d([0], [0], [0])
                com_point.set_color('grey')
                com_point.set_markersize(5)
        else:
            com_point.set_data_3d([], [], [])
            
        # Update CoM Floor point  
        if com_floor_data is not None and frame < len(com_floor_data):
            if com_floor_data[frame, 3] > 0:
                x, y, z = com_floor_data[frame, :3]
                com_floor_point.set_data_3d([x], [y], [z])
                com_floor_point.set_color('blue')
                com_floor_point.set_markersize(10)
            else:
                # Plot grey dot at origin when no valid CoM Floor
                com_floor_point.set_data_3d([0], [0], [0])
                com_floor_point.set_color('grey')
                com_floor_point.set_markersize(5)
        else:
            com_floor_point.set_data_3d([], [], [])
            
        # Update trails (show last 100 valid points only)
        trail_length = min(100, frame + 1)
        start_frame = max(0, frame - trail_length + 1)
        
        # CoM trail
        if com_data is not None:
            trail_frames = range(start_frame, frame + 1)
            trail_com_mask = [(f < len(com_data) and com_data[f, 3] > 0) for f in trail_frames]
            valid_trail_frames = [f for f, valid in zip(trail_frames, trail_com_mask) if valid]
            
            if valid_trail_frames:
                trail_x = com_data[valid_trail_frames, 0]
                trail_y = com_data[valid_trail_frames, 1] 
                trail_z = com_data[valid_trail_frames, 2]
                com_trail.set_data_3d(trail_x, trail_y, trail_z)
            else:
                com_trail.set_data_3d([], [], [])
        
        # Floor trail
        if com_floor_data is not None:
            trail_frames = range(start_frame, frame + 1)
            trail_floor_mask = [(f < len(com_floor_data) and com_floor_data[f, 3] > 0) for f in trail_frames]
            valid_trail_frames = [f for f, valid in zip(trail_frames, trail_floor_mask) if valid]
            
            if valid_trail_frames:
                trail_x = com_floor_data[valid_trail_frames, 0]
                trail_y = com_floor_data[valid_trail_frames, 1]
                trail_z = com_floor_data[valid_trail_frames, 2] 
                floor_trail.set_data_3d(trail_x, trail_y, trail_z)
            else:
                floor_trail.set_data_3d([], [], [])
                
        # Update title with frame info
        ax.set_title(f'3D World - {title} - Frame {frame}/{frames_to_render-1}')
        
        return com_point, com_floor_point, com_trail, floor_trail
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=frames_to_render, 
                        interval=50, blit=False, repeat=True)
    
    # Save as MP4 if requested
    if save_path:
        print(f"Saving animation to {save_path}...")
        # Try FFmpeg first
        anim.save(save_path, writer='ffmpeg', fps=50, bitrate=1800)
        print("Animation saved with FFmpeg!")
    
    return anim

def apply_rz_minus90_transformation(com_data):
    """
    Apply Rz(-90°) rotation to CoM data: (X,Y,Z) -> (Y,-X,Z)
    com_data shape: (N_frames, 4) where columns are [X, Y, Z, confidence]
    Returns: transformed CoM data with same shape
    """
    if com_data is None or len(com_data) == 0:
        return com_data
    
    # Create copy to avoid modifying original data
    com_transformed = com_data.copy()
    
    # Apply transformation: (X,Y,Z) -> (Y,-X,Z)
    com_transformed[:, 0] = com_data[:, 1]   # new X = old Y
    com_transformed[:, 1] = -com_data[:, 0]  # new Y = -old X
    com_transformed[:, 2] = com_data[:, 2]   # new Z = old Z (unchanged)
    # confidence column (index 3) remains unchanged
    
    return com_transformed

# joint order
joint_order_mapping = {
    'RSJC': 'Right Shoulder',
    'REJC': 'Right Elbow',
    'RRJC': 'Right Wrist',
    'LSJC': 'Left Shoulder',
    'LEJC': 'Left Elbow',
    'LRJC': 'Left Wrist',
    'RHJC': 'Right Hip',
    'RKJC': 'Right Knee',
    'RAJC': 'Right Ankle',
    'LHJC': 'Left Hip',
    'LKJC': 'Left Knee',
    'LAJC': 'Left Ankle',
    'PJC': 'Pelvis Center',
    'WJC': 'Waist',
    'NJC': 'Top of Neck',
    'CJC': 'Clavicle',
    'TJC': 'Thorax'
}
joint_order = ['Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 
               'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle', 
               'Pelvis Center', 'Waist', 'Top of Neck', 'Clavicle', 'Thorax']
marker_order = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN', # 15
                'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', #30
                'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']


def get_consolidated_data(jsons, out_dir, apply_transform=True):
    """
    Extract and consolidate CoM, joints, and markers from Vicon JSON files.
    
    Returns:
        data_dict with keys: 'CoM', 'joints', 'markers'
        - CoM: dict with 'CentreOfMass' and 'CentreOfMassFloor' (N, 4)
        - joints: ordered array (N, 17, 4) matching joint_order
        - markers: dict with marker names as keys (N, 4)
    """
    
    for j in jsons:
        print(f"Processing {j}")
        with open(j, 'r') as f:
            data = json.load(f)
      
        om_idx = os.path.basename(j).replace('.json','')
        om_idx_int = int(om_idx.replace("OM", "")) 
        
        # Initialize data structure
        vicon_data = {
            'CoM': {},
            'joints': None,
            'markers': OrderedDict()
        }
       
        # 1. Extract Marker data
        markers = data['Markers']
        # Initialize marker data dict with None values
        for name in marker_order:
            vicon_data['markers'][name] = None
            
        for marker in markers:
            name = marker['name']
            if name in marker_order:
                trajectories = np.array(marker['trajectories']).T  # Shape: (N_frames, 3)
                conf = np.array(marker['valid'])  # Shape: (N_frames,)
                
                if apply_transform:
                    trajectories = apply_rz_minus90_transformation(trajectories)
                
                # Create array shaped (N, 4) - [X, Y, Z, confidence]
                marker_array = np.zeros((trajectories.shape[0], 4))
                marker_array[:, :3] = trajectories
                marker_array[:, 3] = conf
                vicon_data['markers'][name] = marker_array
       
        # Frame count comes from the markers, which are present even for takes
        # that were never run through Plug-in-Gait.
        marker_n_frames = None
        for arr in vicon_data['markers'].values():
            if arr is not None:
                marker_n_frames = arr.shape[0]
                break

        # 2. Extract Model Output (CoM + Joints)
        # A take that was labelled but never run through the Vicon model pipeline
        # has an empty ModelOutput. That is not an error -- the markers, video,
        # pressure and OpenPose for that take are still usable -- so emit what
        # exists and let the caller decide.
        model_output = data.get('ModelOutput', [])

        # Initialize joints array - we'll fill this in the correct order
        n_frames = None
        joint_data_dict = {}
        
        for out in model_output:
            name = out['name']
           
            if name  not in ['CentreOfMass', 'CentreOfMassFloor'] and name not in joint_order_mapping.keys():
                continue
           
            trajectories = np.array(out['data']).T  # Shape: (N_frames, 3)
            conf = np.array(out['valid'])  # Shape: (N_frames,)
            
            if n_frames is None:
                n_frames = trajectories.shape[0]
           
            if apply_transform:
                trajectories = apply_rz_minus90_transformation(trajectories)
            
            # Create array shaped (N, 4) - [X, Y, Z, confidence]
            data_array = np.zeros((n_frames, 4))
            data_array[:, :3] = trajectories
            data_array[:, 3] = conf
            
            # Check if this is CoM data
            if name in ['CentreOfMass', 'CentreOfMassFloor']:
                vicon_data['CoM'][name] = data_array

            # Check if this is joint data
            elif name in joint_order_mapping:
                joint_name = joint_order_mapping[name]
                joint_data_dict[joint_name] = data_array
        
        # 3. Create ordered joint array (N, 17, 4), when there is model output
        has_model_output = n_frames is not None
        if has_model_output:
            vicon_data['joints'] = np.zeros((n_frames, len(joint_order), 4))

            for i, joint_name in enumerate(joint_order):
                if joint_name in joint_data_dict:
                    vicon_data['joints'][:, i, :] = joint_data_dict[joint_name]
        else:
            n_frames = marker_n_frames
            print(f"  [!] {om_idx}: ModelOutput is empty -- no CoM, no joint centres.")
            print(f"      This take was labelled but never run through Plug-in-Gait.")
            print(f"      Writing markers only; re-export from Nexus to get the rest.")

        if n_frames is None:
            print(f"  [!] {om_idx}: no markers and no model output -- nothing to save. Skipping.")
            continue

        # 4. Save data
        base_filename = os.path.splitext(os.path.basename(j))[0]
        out_path = os.path.join(out_dir, base_filename)
        os.makedirs(out_path, exist_ok=True)

        has_com = {'CentreOfMass', 'CentreOfMassFloor'} <= set(vicon_data['CoM'])
        if has_model_output and not has_com:
            print(f"  [!] {om_idx}: model output present but CoM channels missing "
                  f"(found: {sorted(vicon_data['CoM']) or 'none'}). Skipping CoM files.")

        if has_model_output and has_com:
            # # Save CoM data
            com_file_name = os.path.join(out_path, 'CoM.npy')
            np.save(com_file_name, vicon_data['CoM']['CentreOfMass'])
            print(f"Saved CoM data with shape {vicon_data['CoM']['CentreOfMass'].shape}")
            com_floor_file_name = os.path.join(out_path, 'CoM_floor.npy')
            np.save(com_floor_file_name, vicon_data['CoM']['CentreOfMassFloor'])
            print(f"Saved CoM data with shape {vicon_data['CoM']['CentreOfMass'].shape}")

        if has_model_output:
            # # Save joint data
            np.save(os.path.join(out_path, 'MOCAP_3D.npy'), vicon_data['joints'])
            print(f"Saved joints data with shape {vicon_data['joints'].shape}")
        
        # # Save marker data as a N,len(marker_order),4 array
        marker_data_array = np.zeros((n_frames, len(marker_order), 4))
        for i, marker_name in enumerate(marker_order):
            if vicon_data['markers'][marker_name] is not None:
                marker_data_array[:, i, :] = vicon_data['markers'][marker_name]
            else:
                print(f"Marker {marker_name} missing - filled with zeros")
        np.save(os.path.join(out_path, 'MOCAP_MRK.npy'), marker_data_array)
        print(f"Saved mocap marker data with shape {marker_data_array.shape}") 
        

        return vicon_data

import argparse
import glob
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and consolidate Vicon Model Output JSON files")
    parser.add_argument('--person', type=str, default='Kyle', help="Person name (e.g., 'Kyle')")
    parser.add_argument('--raw_data_dir', type=str, default=RAW, help="Directory containing raw data")
    parser.add_argument('--out_dir', type=str, default=WORK, help="Output directory for consolidated data")
    parser.add_argument('--no_world_fix', action='store_true', help="If set, do not apply Rz(-90) transformation to data")
    parser.add_argument('--specific_oms', nargs='+', type=int, default=None, help="List of specific OM indices to process (e.g., --specific_oms 1 2 3)")
    args = parser.parse_args()
     
    person = args.person
    raw_data_dir = args.raw_data_dir
    out_dir = os.path.join(args.out_dir, 'Model_output', person)
    
    os.makedirs(out_dir, exist_ok=True) 
    natural_sort = lambda l: sorted(l, key=lambda s: int(''.join(filter(str.isdigit, s)) or -1))
    json_dir = os.path.join(raw_data_dir, 'Model_output', person)
    jsons = glob.glob(os.path.join(json_dir, '*.json'))
    jsons = natural_sort(jsons)

    # Filter JSON files if specific OM indices are provided
    if args.specific_oms is not None:
        filtered_jsons = []
        for json_file in jsons:
            for om_index in args.specific_oms:
                if f"OM{om_index}" in json_file:
                    filtered_jsons.append(json_file)
                    break
        jsons = filtered_jsons

    # Process all JSON files
    for json_file in jsons:
        data = get_consolidated_data([json_file], out_dir, apply_transform=not args.no_world_fix)