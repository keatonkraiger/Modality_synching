import numpy as np
import pickle
import scipy.io as sio
import json
from scipy.io.matlab import mat_struct
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from glob import glob
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import WORK_BODY25, RAW_XCPS  # noqa: E402

def _check_keys(d):
    """Convert mat_struct objects to dicts."""
    for key in d:
        if isinstance(d[key], mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):
    """Recursively convert mat_struct to nested dict."""
    out = {}
    for field in matobj._fieldnames:
        elem = getattr(matobj, field)
        if isinstance(elem, mat_struct):
            out[field] = _todict(elem)
        elif isinstance(elem, np.ndarray) and elem.dtype == np.object_:
            out[field] = [_todict(e) if isinstance(e, mat_struct) else e for e in elem]
        else:
            out[field] = elem
    return out

def loadmat_struct(filename):
    """Load .mat file and convert MATLAB structs to dicts."""
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def to_serializable(obj):
    """Convert NumPy objects to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

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

def animate_body25(data, n_frames=None, threshold=0., interval=1, save_path=None, azim=89, elev=20):
    """
    Animate BODY25 skeletons from OpenPose data.

    Args:
        data (np.ndarray): shape (N, 25, 4), where last dim is (x,y,z,conf).
        n_frames (int): number of frames to animate (default: all).
        threshold (float): minimum confidence to plot a joint.
        interval (int): delay between frames in ms.
        save_path (str): optional path to save animation (e.g. 'out.mp4').
    """
    assert data.shape[1] == 25 and data.shape[2] == 4, "Data must be (N,25,4)."
    n_total = data.shape[0]
    if n_frames is None:
        n_frames = n_total
    n_frames = min(n_frames, n_total)

    # --- setup figure ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scat = ax.scatter([], [], [], s=30, c="red")
    lines = [ax.plot([], [], [], c="k", linewidth=1)[0] for _ in BODY_PARTS]

    # axis limits fixed across all frames
    xs = data[:, :-1, 0]
    ys = data[:, :-1, 1]
    zs = data[:, :-1, 2]
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(0, zs.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("BODY25 Animation")
    
    # set view angle
    ax.view_init(elev=elev, azim=azim)

    def update(frame):
        joints = data[frame, :-1, :]  # ignore background joint
        xs, ys, zs, conf = joints.T
        mask = conf > threshold

        scat._offsets3d = (xs[mask], ys[mask], zs[mask])

        for line, (i, j) in zip(lines, BODY_PARTS):
            if i < 24 and j < 24 and conf[i] > threshold and conf[j] > threshold:
                line.set_data([xs[i], xs[j]], [ys[i], ys[j]])
                line.set_3d_properties([zs[i], zs[j]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])
        # Set title to show frame number
        ax.set_title(f"BODY25 Animation - Frame {frame+1}/{n_frames}")

        return [scat, *lines]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        anim.save(save_path, fps=1000//interval)
    else:
        plt.show()

    return anim

class Camera:
    """
    Triangulation-ready camera.
    - Stores raw pose (position, quaternion [x,y,z,w])
    - Optionally applies world-frame fix on init
    - Builds R, t, K, P the CaraPost way:  P = K [ R | -R*C ]
    """
    def __init__(self, position, orientation_xyzw, device_id, user_id,
                 focal_length, pixel_aspect_ratio=1.0, skew=0.0,
                 principal_point=(0.0, 0.0), radial=None,
                 apply_world_fix=True):
        # Raw pose
        self.device_id = str(device_id)
        self.user_id   = int(user_id)
        self.position  = np.array(position, dtype=float)
        self.quat      = normalize_quat(np.array(orientation_xyzw, dtype=float))  # [x,y,z,w]

        # Intrinsics
        self.foclen       = float(focal_length)           # f
        self.aspectratio  = float(pixel_aspect_ratio)     # a
        self.skew         = float(skew)                   # k
        self.ppx, self.ppy = map(float, principal_point)  # (x_pp, y_pp)
        self.radial       = np.array(radial if radial is not None else [0,0,0], dtype=float)
        self.pre_fix_position = self.position.copy()
        self.pre_fix_orientation     = self.quat.copy()
        
        # Apply world-frame fix (your Rz(-90°) correction)
        if apply_world_fix:
            print(f"Applying world-frame fix to camera {self.device_id}")
            self.position, self.quat = apply_world_rotation_to_pose(self.position, self.quat)
           
        dev_to_name = {'2111392': 'Camera 1','2111394': 'Camera 2'} 
        x, y, z, w = self.quat
        if self.device_id == '2111392':
            x=-x
            y = -y
            z=-z
            w = -w
            self.quat = np.array([x, y, z, w], dtype=float) 
            
        self.orientation = self.quat  # alias 
        # Build matrices
        self.Rmat = quat_to_R_xyzw(self.quat, convention='opencv')             # world -> camera
        self.tvec = -self.Rmat @ self.position            # -R*C
        self.Kmat = self._build_K()                       # CaraPost K
        self.Pmat =  self.Pmat = np.hstack([self.Rmat, self.tvec.reshape(3,1)]) #self._build_P()                       # K [R | t]

    def _build_K(self):
        """Build intrinsic matrix K in standard form."""
        fx = self.foclen
        fy = self.foclen * self.aspectratio  # or self.foclen / self.aspectratio depending on convention
        
        return np.array([
            [fx,   self.skew, self.ppx],
            [0.0,  fy,        self.ppy],
            [0.0,  0.0,       1.0]
        ], dtype=float)

    def _build_P(self):
        Rt = np.hstack([self.Rmat, self.tvec.reshape(3,1)])
        return self.Kmat @ Rt

    def to_triangulation_dict(self):
        """Matches your triangulator expectations."""
        (['orientation', 'position', 'prinpoint', 'radial', 'aspectratio', 'skew', 'Pmat', 'Rmat', 'Kmat'])
        return {
            "Pmat": self.Pmat.copy(),
            "Kmat": self.Kmat.copy(),
            "Rmat": self.Rmat.copy(),
            "foclen": self.foclen,
            "orientation": self.quat.copy(),
            "pre_fix_orientation": self.pre_fix_orientation.copy(),
            "pre_fix_position": self.pre_fix_position.copy(),
            "prinpoint": np.array([self.ppx, self.ppy], dtype=float),
            "skew": self.skew,
            "aspectratio": self.aspectratio,
            "radial": self.radial.copy(),
            "position": self.position.copy(),
        }
        
def rescale_K(K, orig_size, new_size):
    """Rescale camera intrinsics from original calibration size to a new image size."""
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]

    K_rescaled = K.copy()
    K_rescaled[0,0] *= sx   # fx
    K_rescaled[1,1] *= sy   # fy
    K_rescaled[0,2] *= sx   # cx
    K_rescaled[1,2] *= sy   # cy
    return K_rescaled

def normalize_P(P):
    """Normalize projection matrix for comparison (scale invariant)."""
    P = np.array(P, dtype=float)
    if P[-1, -1] != 0:
        return P / P[-1, -1]
    else:
        return P / np.linalg.norm(P)
        
# ---------------- Quaternion utils ([x,y,z,w], w last) ----------------
def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def quat_conj(q):  # inverse for unit quats
    x,y,z,w = q
    return np.array([-x, -y, -z,  w], dtype=float)

def quaternion_multiply(q1, q2):
    """Hamilton product for [x,y,z,w] quats: q = q1 ⊗ q2"""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_to_R_xyzw(q, convention="opencv"):
    x,y,z,w = normalize_quat(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy+zz),   2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=float)

    if convention == "vicon":
        R = R.T   # transpose fixes to match MATLAB CaraPost export

    return R

# World-frame fix: (X,Y,Z)->(Y,-X,Z) = Rz(-90°)
Q_WORLD_FIX = normalize_quat(np.array([0.0, 0.0, -np.sin(np.pi/4), np.cos(np.pi/4)], dtype=float))
def apply_world_rotation_to_pose(pos_xyz, quat_xyzw):
    """Apply world-frame rotation to camera pose."""
    quat_xyzw = normalize_quat(quat_xyzw)
    q_new  = quaternion_multiply(quat_xyzw, quat_conj(Q_WORLD_FIX))  # R' = R * Rs^T
    x,y,z  = pos_xyz
    pos_new = np.array([y, -x, z], dtype=float)                      # C' = Rs * C
    return pos_new, normalize_quat(q_new)
    
# ---------- XCP parsing ----------
def _get_float(attr_dict, key, default=0.0):
    v = attr_dict.get(key)
    if v is None: return default
    try:
        return float(str(v).split()[0])
    except Exception:
        return default

def _get_floats(attr_dict, key, n=None, default=None):
    v = attr_dict.get(key)
    if v is None:
        return default if default is not None else []
    try:
        arr = [float(x) for x in str(v).split()]
        if n is not None:
            if len(arr) >= n: return arr[:n]
            # pad
            return arr + [0.0]*(n-len(arr))
        return arr
    except Exception:
        return default if default is not None else []
    
def parse_vue_cameras_as_objects(xcp_path, apply_world_fix=True, scale_position=1.0):
    """
    Returns dict: device_id -> Camera instance
    (robust to a few attribute naming variations)
    """
    tree = ET.parse(xcp_path)
    root = tree.getroot()
    cams = {}

    for cam in root.findall(".//Camera"):
        if cam.attrib.get("DISPLAY_TYPE") != "Vue":
            continue

        user_id   = int(cam.attrib.get("USERID"))
        device_id = cam.attrib.get("DEVICEID")

        # Per docs, pixel aspect ratio is on the Camera element
        pixel_aspect_ratio = _get_float(cam.attrib, "PIXEL_ASPECT_RATIO")

        # KeyFrame holds pose + lens params
        kf = cam.find(".//KeyFrame")
        if kf is None:
            continue

        # Pose
        pos = np.array(_get_floats(kf.attrib, "POSITION", n=3), dtype=float) * scale_position  # [x,y,z]
        q   = np.array(_get_floats(kf.attrib, "ORIENTATION", n=4), dtype=float)  # [x,y,z,w]

        # Intrinsics-ish fields (names vary a bit; default to safe values)
        f   = _get_float(kf.attrib, "FOCAL_LENGTH")
        k   = _get_float(kf.attrib, "SKEW")  # often 0
        pp  = _get_floats(kf.attrib, "PRINCIPAL_POINT")
        rad = _get_floats(kf.attrib, "VICON_RADIAL")
        if len(rad) < 3:  # pad to k1,k2,k3
            rad = (rad + [0.0]*3)[:3]

        cams[device_id] = Camera(
            position=pos, orientation_xyzw=q,
            device_id=device_id, user_id=user_id,
            focal_length=f, pixel_aspect_ratio=pixel_aspect_ratio,
            skew=k, principal_point=pp, radial=rad,
            apply_world_fix=apply_world_fix
        )

    return cams

def perform_radial_correction(x_distorted, y_distorted, radial_params):
    radk = np.zeros(3, dtype=float)
    radk[:len(radial_params)] = radial_params

    r2val = x_distorted**2 + y_distorted**2
    rtmp = 1 + r2val * (radk[0] + r2val * (radk[1] + r2val * radk[2]))

    filmx = x_distorted * rtmp
    filmy = y_distorted * rtmp

    return filmx, filmy

def triangulate_body25_with_distortion(
    ux, uy, ux_2, uy_2, cam1, cam2, front_conf, side_conf, thresh
):
    R1 = cam1["Pmat"][:, :-1]
    t1 = cam1["Pmat"][:, -1]
    K1 = cam1["Kmat"]

    R2 = cam2["Pmat"][:, :-1]
    t2 = cam2["Pmat"][:, -1]
    K2 = cam2["Kmat"]

    c1 = -R1.T @ t1
    c2 = -R2.T @ t2

    K1_tmp = K1.copy()
    K2_tmp = K2.copy()

    K1_tmp[1, 1] = cam1["aspectratio"]
    K1_tmp[0, 0] = 1
    K2_tmp[1, 1] = cam2["aspectratio"]
    K2_tmp[0, 0] = 1

    u1_tmp = np.linalg.inv(K1_tmp) @ np.vstack([ux, uy, np.ones(len(ux))])
    u2_tmp = np.linalg.inv(K2_tmp) @ np.vstack([ux_2, uy_2, np.ones(len(ux_2))])

    u1_tmp[0, :], u1_tmp[1, :] = perform_radial_correction(
        u1_tmp[0, :], u1_tmp[1, :], cam1["radial"]
    )
    u2_tmp[0, :], u2_tmp[1, :] = perform_radial_correction(
        u2_tmp[0, :], u2_tmp[1, :], cam2["radial"]
    )

    u1_tmp[2, :] = cam1["foclen"]
    u2_tmp[2, :] = cam2["foclen"]

    u1 = R1.T @ u1_tmp
    u2 = R2.T @ u2_tmp

    u1_normalized = np.zeros_like(u1)
    u2_normalized = np.zeros_like(u2)
    for i in range(len(ux)):
        u1_normalized[:, i] = u1[:, i] / np.linalg.norm(u1[:, i])
        u2_normalized[:, i] = u2[:, i] / np.linalg.norm(u2[:, i])

    points = np.zeros((3, len(ux)))
    I = np.eye(3)

    for i in range(len(ux)):
        if front_conf[i] <= thresh or side_conf[i] <= thresh:
            P = np.zeros(3)
        else:
            total_conf = front_conf[i] + side_conf[i]
            norm_front = front_conf[i] / total_conf
            norm_side = side_conf[i] / total_conf

            P_term = (
                norm_front * (I - np.outer(u1_normalized[:, i], u1_normalized[:, i]))
                + norm_side * (I - np.outer(u2_normalized[:, i], u2_normalized[:, i]))
            )

            c1_term = (
                norm_front * (I - np.outer(u1_normalized[:, i], u1_normalized[:, i]))
            ) @ c1
            c2_term = (
                norm_side * (I - np.outer(u2_normalized[:, i], u2_normalized[:, i]))
            ) @ c2
            c_term = c1_term + c2_term

            if np.linalg.cond(P_term) > 1e12:
                P = np.zeros(3)
            else:
                P = np.linalg.solve(P_term, c_term)
        points[:, i] = P
    return points

def enforce_ground_plane(points3d, translation_cap=50):
    """
    Shift all 3D points up so the minimum z is 0,
    capped to avoid over-correcting from outliers.

    Args:
        points3d : (N, J, 3) or (3, J) ndarray
            Triangulated 3D points
        translation_cap : float
            Maximum allowed upward translation (meters).
            Prevents rogue points from pulling the floor too low.
    """
    pts = np.asarray(points3d)
    z_min = np.nanmin(pts[..., 2])  # handle NaNs if you zero bad joints
    
    # Cap the correction
    dz = min(max(-z_min, 0.0), translation_cap)
    pts[..., 2] += dz
    return pts 
    
    
def perform_triangulation(pose_v1, pose_v2, vue1, vue2, conf_thresh=0.1, scale_x=1.0, scale_y=1.0, encorce_positive_z=True): 
    num_frames, num_joints, _ = pose_v1.shape
    triangulated = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
                            
    for f in range(num_frames):
        if f % 1000 == 0:
            print(f"Processing frame {f+1}/{num_frames}")

        v1_xy = pose_v1[f, :, :2].T  # shape [2, 25]
        v2_xy = pose_v2[f, :, :2].T
        v1_conf = pose_v1[f, :, 2]
        v2_conf = pose_v2[f, :, 2]

        pts3d = triangulate_body25_with_distortion(
            v1_xy[0] * scale_x, v1_xy[1] * scale_y,
            v2_xy[0] * scale_x, v2_xy[1] * scale_y,
            vue1, vue2,
            v1_conf, v2_conf,
            thresh=conf_thresh
        )
            
        triangulated[f, :, :] = pts3d.T
        

    if encorce_positive_z:
        triangulated = enforce_ground_plane(triangulated)
        
    # Confidence averaging
    combined_conf = np.sqrt(pose_v1[:, :, 2] * pose_v2[:, :, 2])
    pose_3d = np.zeros((num_frames, num_joints, 4))
    pose_3d[:, :, :3] = triangulated
    pose_3d[:, :, 3] = combined_conf

    # Handle invalids
    failed = np.all(triangulated == 0, axis=2) | (combined_conf == 0)
    pose_3d[failed] = 0
    return pose_3d

image_height = 1080
image_width = 1920
vue1_file = '/mnt/d/Data/PSU100/Modality_wise/Camera_Parameter/Subject1/Parameters_V1_1.mat'
vue2_file = '/mnt/d/Data/PSU100/Modality_wise/Camera_Parameter/Subject1/Parameters_V2_1.mat' 
print_comparison = False
# with open('vue1.pkl', 'rb') as f:
#     correct_vue1 = pickle.load(f)
# with open('vue2.pkl', 'rb') as f:
#     correct_vue2 = pickle.load(f)
      

import argparse 
def main():
    parser = argparse.ArgumentParser(description="Triangulate BODY25 3D poses from 2D OpenPose data")
    parser.add_argument("--body25_dir", type=str, default=WORK_BODY25, help="Directory containing BODY25 NPY files and videos")
    parser.add_argument("--xcp_dir", type=str, default=RAW_XCPS, help="Directory containing camera parameter XCP files")
    parser.add_argument('--subjects', type=str, nargs='+', required=True, help="List of subjects to process")
    parser.add_argument('--animate', action='store_true', help="Whether to animate the 3D poses")
    parser.add_argument('--specific_oms', type=str, nargs='+', default=[], help="Specific OM takes to process (e.g., OM1, OM2)")
    parser.add_argument('--dry_run', action='store_true', help="If set, do not save any files")
    parser.add_argument('--no_world_fix', action='store_true', help="If set, do not apply world-frame fix to camera poses")
    args = parser.parse_args() 
    

    print_out = ['orientation', 'position', 'Pmat', 'Kmat', 'Rmat']    
    dev_to_name = {'2111392': 'Camera 1','2111394': 'Camera 2'} 
    subject_broken_takes = {
        'Spencer': ['OM1',],
        'Abhinav': ['OM11'],
        'Varad': ['OM1', 'OM3', 'OM8', 'OM9', 'OM10', 'OM12'] 
    }
    for subject in args.subjects:
        # Get camera parameters
        xcp_files = glob(os.path.join(args.xcp_dir, subject, f'OM*.xcp'))
        xcp_files.sort()
        
        # new_camera_params_dir = os.path.join(args.xcp_dir, subject, 'Cleaned')
        # os.makedirs(new_camera_params_dir, exist_ok=True)
        
        for xcp in xcp_files:
            if args.specific_oms and not any(om in xcp for om in args.specific_oms):
                print(f"Skipping {xcp} as it's not in the specific_oms list.")
                continue
                
            
            om = os.path.basename(xcp).replace('.xcp', '')
            print(f"Processing {subject} {om}")
            cams = parse_vue_cameras_as_objects(xcp, apply_world_fix= not args.no_world_fix,)
            vue1 = cams['2111392'].to_triangulation_dict()
            vue2 = cams['2111394'].to_triangulation_dict()

            broken_oms = subject_broken_takes.get(subject, [])
            if om not in broken_oms:
                scale_x = 1
                scale_y = 1
            else:
                print(f" {om} has non-standard resolution, applying scaling.")
                scale_x, scale_y = 1,1
                vue1['Kmat'] = rescale_K(vue1['Kmat'], (image_width, image_height), (1280, 720),)
                vue2['Kmat'] = rescale_K(vue2['Kmat'],  (image_width, image_height), (1280, 720),)

            # Load 2D poses
            pose_path = os.path.join(args.body25_dir, subject, om)
            # Check if pose path exists
            if not os.path.exists(os.path.join(pose_path, 'BODY25_V1.npy')) or not os.path.exists(os.path.join(pose_path, 'BODY25_V2.npy')):
                print(f"Pose path {pose_path} does not exist. Skipping.")
                continue
            pose_v1 = np.load(os.path.join(pose_path, 'BODY25_V1.npy'))  # (N, 25, 3)
            pose_v2 = np.load(os.path.join(pose_path, 'BODY25_V2.npy'))  # (N, 25, 3)
  
            
            pose_3d = perform_triangulation(pose_v1, pose_v2, vue1, vue2, conf_thresh=0.0, scale_x=scale_x, scale_y=scale_y, encorce_positive_z=False)
            if not args.dry_run:
                np.save(os.path.join(pose_path, 'BODY25_3D.npy'), pose_3d)

            if args.animate:
                animate_body25(pose_3d, n_frames=1000, threshold=0.0, interval=50, azim=91, elev=20)
            
        
if __name__ == "__main__":
    main()