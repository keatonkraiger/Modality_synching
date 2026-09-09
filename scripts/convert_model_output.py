import os
import json
import numpy as np
import argparse
import cv2
from pathlib import Path
from collections import OrderedDict

# -------------------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------------------

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def apply_rz_minus90(data_array):
    """
    Applies Rz(-90) rotation: (X, Y, Z) -> (Y, -X, Z).
    Only valid for spatial position vectors (X,Y,Z).
    """
    data = data_array.copy()
    
    # Shape (N, 3) or (N, 4) where cols are X,Y,Z,Valid
    if data.ndim == 2 and data.shape[1] >= 3:
        x = data[:, 0].copy()
        y = data[:, 1].copy()
        data[:, 0] = y
        data[:, 1] = -x
        
    return data

def upsample_linear(data):
    """Upsample by 2x via linear interpolation."""
    if data.ndim == 1:
        upsampled = np.empty(2 * len(data), dtype=data.dtype)
        upsampled[0::2] = data
        upsampled[1:-1:2] = (data[:-1] + data[1:]) / 2
        upsampled[-1] = data[-1]
    else:
        new_shape = list(data.shape)
        new_shape[0] *= 2
        upsampled = np.empty(new_shape, dtype=data.dtype)
        upsampled[0::2] = data
        upsampled[1:-1:2] = (data[:-1] + data[1:]) / 2
        upsampled[-1] = data[-1]
    return upsampled

# -------------------------------------------------------------------------
# 2. Extraction (Get Everything Raw)
# -------------------------------------------------------------------------

def get_segment_data(raw_data_dict):
    segment_data = OrderedDict()
    if 'Segments' in raw_data_dict:
        for segment in raw_data_dict['Segments']:
            name = segment['name']
            segment_data[name] = {
                'group': segment.get('parent', None),
                'children': segment.get('children', []),
                'markers': segment.get('markers', [])
            }
    return segment_data

def get_raw_content(json_path, apply_transform=True):
    with open(json_path, 'r') as f:
        raw = json.load(f)

    # 1. Parameters & Segments
    params = raw.get('Params', {})
    segments = get_segment_data(raw)

    # 2. Model Outputs
    model_outputs = OrderedDict()
    if 'ModelOutput' in raw:
        for model in raw['ModelOutput']:
            name = model['name']
            data = np.array(model['data']).T 
            valid = np.array(model['valid'])
            
            if apply_transform and data.shape[1] >= 3:
                # Safe Transform Check
                comp_names = model.get('componentNames', [])
                comp_types = model.get('componentTypes', [])
                is_xyz = (comp_names == ['X', 'Y', 'Z'])
                allowed_types = {'Length', 'Position', 'Translation', 'Point', 'Vector'}
                is_spatial = all(t in allowed_types for t in comp_types) if comp_types else False

                if is_xyz and is_spatial:
                    data = apply_rz_minus90(data)
                elif name in ['CentreOfMass', 'CentreOfMassFloor']:
                    data = apply_rz_minus90(data)

            model_outputs[name] = {
                'group': model.get('group', ''),
                'component_names': model.get('componentNames', []),
                'component_types': model.get('componentTypes', []),
                'data': data,
                'valid': valid
            }

    # 3. Markers
    markers = OrderedDict()
    if 'Markers' in raw:
        for marker in raw['Markers']:
            name = marker['name']
            traj = np.array(marker['trajectories']).T
            valid = np.array(marker['valid'])
            
            if apply_transform:
                traj = apply_rz_minus90(traj)

            combined = np.zeros((traj.shape[0], 4))
            combined[:, :3] = traj
            combined[:, 3] = valid
            markers[name] = combined

    return {
        'params': params,
        'segments': segments,
        'model_outputs': model_outputs,
        'markers': markers
    }

# -------------------------------------------------------------------------
# 3. Processing Helpers (Resample -> Slice)
# -------------------------------------------------------------------------

def resample_structure(raw_content, mode):
    """
    Applies downsampling or upsampling to the entire structure.
    Does NOT slice or trim time yet.
    """
    if mode is None:
        return raw_content

    resampled = {
        'params': raw_content['params'],
        'segments': raw_content['segments'],
        'model_outputs': OrderedDict(),
        'markers': OrderedDict()
    }

    def process_arr(arr):
        if mode == 'downsample':
            return arr[::2]
        elif mode == 'upsample':
            return upsample_linear(arr)
        return arr

    for name, content in raw_content['model_outputs'].items():
        resampled['model_outputs'][name] = {
            'group': content['group'],
            'component_names': content['component_names'],
            'component_types': content['component_types'],
            'data': process_arr(content['data']),
            'valid': process_arr(content['valid'])
        }

    for name, data in raw_content['markers'].items():
        resampled['markers'][name] = process_arr(data)

    return resampled

def slice_structure(content, start, end):
    """
    Slices arrays from [start:end].
    Returns the final structured dictionary.
    """
    final_data = {
        'params': content['params'],
        'segments': content['segments'],
        'model_output': OrderedDict(),
        'markers': OrderedDict()
    }

    def safe_slice(arr):
        # Clamp indices to handle potential edge cases (though start/end should be valid)
        if start >= arr.shape[0]:
            return np.empty((0,) + arr.shape[1:])
        actual_end = min(end, arr.shape[0])
        return arr[start:actual_end]

    for name, c in content['model_outputs'].items():
        final_data['model_output'][name] = {
            'group': c['group'],
            'component_names': c['component_names'],
            'component_types': c['component_types'],
            'data': safe_slice(c['data']),
            'valid': safe_slice(c['valid'])
        }

    for name, d in content['markers'].items():
        final_data['markers'][name] = safe_slice(d)

    return final_data

# -------------------------------------------------------------------------
# 4. Main Process
# -------------------------------------------------------------------------

def process_subject(raw_data_dir, untrimmed_dir, name, save_root, offsets_dir, specific_oms=None,
                    apply_transform=True):
    
    # Setup Paths
    raw_json_dir = Path(raw_data_dir) / "Model_output" / name
    body25_dir = Path(untrimmed_dir) / "BODY25" / name
    video_dir = Path(raw_data_dir) / "Video" / name
    pressure_dir = Path(untrimmed_dir) / "Pressure" / name
    
    if not body25_dir.exists():
        print(f"  [Error] BODY25 directory not found at {body25_dir}. Skipping subject.")
        return

    # Find OMs
    om_dirs = sorted([d for d in body25_dir.iterdir() if d.is_dir() and d.name.startswith("OM")])

    for om_dir in om_dirs:
        om_idx = om_dir.name
        om_idx_int = int(om_idx.replace("OM", ""))
        
        if specific_oms is not None and om_idx_int not in specific_oms:
            continue

        print(f"\nProcessing {name} - {om_idx} ...")
        
        # --- 1. Load Offset Config ---
        offset_file = Path(offsets_dir) / name / f"{om_idx}.json"
        if not offset_file.exists():
             offset_file = Path(offsets_dir) / name / f"{om_idx}_cfg.json"
             
        if not offset_file.exists():
            print("  Skipping: No offset file found.")
            continue
            
        with open(offset_file, "r") as f:
            config = json.load(f)
        offset = config["offset"]
        original_video_fps = config.get("original_video_fps", 50.0)
        
        print(f"  Offset: {offset}")
        print(f"  Original video fps: {original_video_fps}")
        
        # --- 2. Validation (Video & BODY25) ---
        v1 = video_dir / f"{om_idx}_V1.mp4"
        v2 = video_dir / f"{om_idx}_V2.mp4"
        
        if not v1.exists() or not v2.exists():
            print(f"  [Error] Videos not found for {om_idx}")
            continue

        b1 = np.load(body25_dir / om_idx / "BODY25_V1.npy")
        b2 = np.load(body25_dir / om_idx / "BODY25_V2.npy")
        b3d = np.load(body25_dir / om_idx / "BODY25_3D.npy")
        
        v1_len, _ = get_video_info(v1)
        v2_len, _ = get_video_info(v2)
        
        print(f"  V1: {v1_len}, V2: {v2_len}, B3D: {b3d.shape[0]}")
        
        lengths = [v1_len, v2_len, b1.shape[0], b2.shape[0], b3d.shape[0]]
        if max(lengths) - min(lengths) > 1:
            print(f"  [Error] Length mismatch > 1 frame. Max diff: {max(lengths)-min(lengths)}")
            # continue # Optionally continue or raise error based on preference
        
        target_len = min(lengths)
        print(f"  Target Length (Video base): {target_len}")

        # --- 3. Load Raw Vicon ---
        json_cands = list(raw_json_dir.glob(f"*{om_idx}*.json"))
        if not json_cands:
            json_cands = list(raw_json_dir.glob(f"*take_{om_idx_int:03d}*.json"))
            
        if not json_cands:
            print("  Skipping: No JSON file found.")
            continue
            
        # Extract RAW (Get everything). apply_transform is the caller's
        # --no_world_fix decision: Rz(-90) belongs on the Summer-25 subjects but
        # not necessarily on earlier sessions, so it must not be hardcoded here.
        raw_content = get_raw_content(json_cands[0], apply_transform=apply_transform)
        
        # --- 4. Determine Resampling Strategy ---
        if not raw_content['markers']:
            print("  Skipping: No markers.")
            continue
            
        vicon_len = list(raw_content['markers'].values())[0].shape[0]
        resample_mode = None
        
        if abs(vicon_len - target_len) <= 2: # Tolerance
            resample_mode = None
            print(f"  Resampling: None ({vicon_len} -> {target_len})")
        elif abs((vicon_len // 2) - target_len) <= 2:
            resample_mode = 'downsample'
            print(f"  Resampling: Downsample ({vicon_len} -> ~{target_len})")
        elif abs((vicon_len * 2) - target_len) <= 2:
            resample_mode = 'upsample'
            print(f"  Resampling: Upsample ({vicon_len} -> ~{target_len})")
        else:
            print(f"  [Error] Large length mismatch: {vicon_len} vs {target_len}")
            continue

        # --- 5. Apply Resampling ONLY ---
        resampled_content = resample_structure(raw_content, resample_mode)
        
        # Check available length after resampling
        # Just grab CoM or a marker to check length
        if 'CentreOfMass' not in resampled_content['model_outputs']:
            n_mo = len(resampled_content['model_outputs'])
            print(f"  Skipping {om_idx}: no CentreOfMass in model output "
                  f"({n_mo} model output entries).")
            print(f"    This take was never run through Plug-in-Gait, so there are no")
            print(f"    segment poses for the instantaneous-CoM stage to use.")
            print(f"    Re-export it from Nexus with the model applied to include it.")
            continue
        available_len = len(resampled_content['model_outputs']['CentreOfMass']['data'])
        others_available = min(target_len, available_len)
        
        # --- 6. Load Pressure & Calculate Sync Indices ---
        pressure_file = pressure_dir / om_idx / "Original_Pressure.npy"
        if not pressure_file.exists():
            print("  Skipping: No pressure file found.")
            continue
            
        pressure = np.load(pressure_file)
        print(f"  Pressure Length: {pressure.shape[0]}")
        
        # The logic you provided:
        if offset < 0:
            others_start = abs(offset)
            pressure_start = 0
            print(f"  -> Video/Vicon started {abs(offset)} frames BEFORE pressure.")
            print(f"  -> Trimming first {others_start} frames of Video/Vicon.")
        else:
            others_start = 0
            pressure_start = offset
            print(f"  -> Pressure started {offset} frames BEFORE video.")
            
        others_remaining = others_available - others_start
        pressure_remaining = pressure.shape[0] - pressure_start
        
        # Determine strict final length limited by whichever stream ends first
        final_len = min(others_remaining, pressure_remaining)
        
        if final_len <= 0:
            print(f"  [Error] Final length is {final_len}. Offset issue?")
            continue
            
        others_end = others_start + final_len
        print(f"  Slice Range: [{others_start}:{others_end}] ({final_len} frames)")
       
        
        # --- 7. Apply Slicing ---
        final_data = slice_structure(resampled_content, others_start, others_end)
        take_com = final_data['model_output']['CentreOfMass']['data']
       
        comp_com_path = Path('Complete') / name / om_idx / 'CoM.npy'
        if comp_com_path.exists():
            comp_com = np.load(comp_com_path)
            assert len(take_com) == len(comp_com), f"Length mismatch in CoM for {name} {om_idx}"
            assert np.allclose(take_com, comp_com[:,:-1], atol=1e-4), f"Data mismatch in CoM for {name} {om_idx}"
            print(f"  CoM cross-check against {comp_com_path}: OK")
        else:
            print(f"  [!] No {comp_com_path} to cross-check against -- run combine_all.py first")
            print(f"      if you want this stage verified against the synced output.")
         
        # --- 8. Save ---
        save_dir = Path(save_root) / name 
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{om_idx}.npy"
       
        print(f"  Saving to {out_file}...")
        np.save(out_file, final_data)
        print("  Done.")

# -------------------------------------------------------------------------
# 5. Entry Point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--persons', nargs='+', type=str, default=['Keaton', 'Kyle', 'Varad', 'Abhinav', 'Spencer'], help="Person name(s)")
    parser.add_argument('--raw_data_dir', type=str, default='raw_data', help="Directory containing raw data")
    parser.add_argument('--out_dir', type=str, default='Model_output', help="Output directory")
    parser.add_argument('--no_world_fix', action='store_true', help="Do not apply Rz(-90)")
    parser.add_argument('--specific_oms', nargs='+', type=int, default=None)
    args = parser.parse_args()

    for person in args.persons:
        print(f"\n{'='*40}")
        print(f"Starting: {person}")
        print(f"{'='*40}")

        process_subject(
            raw_data_dir=args.raw_data_dir,
            untrimmed_dir='untrimmed', # Assumed based on previous context
            name=person,
            save_root=args.out_dir,
            offsets_dir='raw_data/offsets',
            specific_oms=args.specific_oms,
            apply_transform=not args.no_world_fix
        )
        
