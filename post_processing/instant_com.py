import argparse
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import VICON_FULL, COMPLETE, COM_COMPARISON  # noqa: E402
import takes  # noqa: E402
from glob import glob

import numpy as np
from tqdm import tqdm


class BaseCoMCalculator:
    def __init__(self, npy_file_path):
        self.npy_file_path = npy_file_path
        self.take_data = np.load(npy_file_path, allow_pickle=True).item()

        self.model_outputs = self.take_data["model_output"]
        self.markers_dict = self.take_data["markers"]

        self.marker_array, self.marker_names = self._normalize_markers(self.markers_dict)
        self.marker_map = {name: i for i, name in enumerate(self.marker_names)}

        self.num_frames = self.marker_array.shape[0]
        self.marker_radius = 14.0 / 2.0

        self.weights = {
            "pelvis": 0.142,
            "femur": 0.1,
            "tibia": 0.0465,
            "foot": 0.0145,
            "humerus": 0.028,
            "radius": 0.016,
            "hand": 0.006,
            "thorax": 0.355,
            "head": 0.081,
        }

        self.dist_ratios = {
            "pelvis": 0.895,
            "femur": 0.567,
            "tibia": 0.567,
            "foot": 0.5,
            "humerus": 0.564,
            "radius": 0.57,
            "hand": 0.6205,
            "thorax": (1 - 0.63),
            "head": 0.52,
        }

    def _normalize_markers(self, markers_obj):
        # Supports both:
        # 1) {'data': (N,M,4), 'marker_names': [...]}
        # 2) OrderedDict({'C7': (N,4), 'RTOE': (N,4), ...})
        if isinstance(markers_obj, dict) and "data" in markers_obj and "marker_names" in markers_obj:
            return markers_obj["data"], list(markers_obj["marker_names"])

        if hasattr(markers_obj, "keys"):
            marker_names = list(markers_obj.keys())
            marker_arrays = [np.asarray(markers_obj[name]) for name in marker_names]
            marker_array = np.stack(marker_arrays, axis=1)  # (N, M, 4)
            return marker_array, marker_names

        raise ValueError(f"Unsupported marker format in {self.npy_file_path}")

    def get_model_data(self, name, frame):
        data = self.model_outputs[name]["data"][frame, :]
        valid_arr = self.model_outputs[name].get("valid", None)
        valid = True if valid_arr is None else bool(valid_arr[frame] > 0.5)
        return data, valid

    def get_marker_data(self, name, frame):
        idx = self.marker_map[name]
        data_full = self.marker_array[frame, idx, :]
        valid = bool(data_full[3] > 0.5) if data_full.shape[0] >= 4 else True
        return data_full[:3], valid

    def bone_segment(self, vicon_rts, axis_index=2):
        distal = vicon_rts[3:6]

        ax, ay, az = vicon_rts[0], vicon_rts[1], vicon_rts[2]
        theta = np.sqrt(ax * ax + ay * ay + az * az)

        R = np.eye(3)
        if theta >= np.finfo(float).eps:
            x, y, z = ax / theta, ay / theta, az / theta
            theta_rad = theta * np.pi / 180.0
            c = np.cos(theta_rad)
            s = np.sin(theta_rad)

            R[0, 0] = c + (1 - c) * x * x
            R[0, 1] = (1 - c) * x * y + s * (-z)
            R[0, 2] = (1 - c) * x * z + s * y
            R[1, 0] = (1 - c) * y * x + s * z
            R[1, 1] = c + (1 - c) * y * y
            R[1, 2] = (1 - c) * y * z + s * (-x)
            R[2, 0] = (1 - c) * z * x + s * (-y)
            R[2, 1] = (1 - c) * z * y + s * x
            R[2, 2] = c + (1 - c) * z * z

        scale = np.mean(vicon_rts[6:9])
        proximal = distal + scale * R[:, axis_index]
        return distal, proximal, R, scale


class InstCoMCalculator(BaseCoMCalculator):
    def process_sequence(self, step=1):
        com_results = []

        for f in tqdm(range(0, self.num_frames, step), desc="Inst CoM Calc", leave=False):
            com_accum = np.zeros(3)
            frame_valid = True

            def add_limb(side, seg, w_key):
                nonlocal frame_valid
                data, valid = self.get_model_data(side + seg, f)
                if not valid:
                    frame_valid = False
                dd, pp, _, _ = self.bone_segment(data)
                return self.weights[w_key] * (dd + self.dist_ratios[w_key] * (pp - dd))

            def add_joint(p_name, d_name, w_key):
                nonlocal frame_valid
                prox, v1 = self.get_model_data(p_name, f)
                dist, v2 = self.get_model_data(d_name, f)
                if not (v1 and v2):
                    frame_valid = False
                return self.weights[w_key] * (dist[:3] + self.dist_ratios[w_key] * (prox[:3] - dist[:3]))

            def add_foot(side):
                nonlocal frame_valid
                seg, v1 = self.get_model_data(side + "FO", f)
                toe, v2 = self.get_marker_data(side + "TOE", f)
                heel, v3 = self.get_marker_data(side + "HEE", f)
                ajc, v4 = self.get_model_data(side + "AJC", f)

                if not (v1 and v2 and v3 and v4):
                    frame_valid = False

                _, _, R, _ = self.bone_segment(seg)
                scale = np.linalg.norm(heel - toe) - self.marker_radius
                return self.weights["foot"] * (ajc[:3] - self.dist_ratios["foot"] * scale * R[:, 2])

            com_accum += add_limb("R", "FE", "femur") + add_limb("L", "FE", "femur")
            com_accum += add_joint("RKJC", "RAJC", "tibia") + add_joint("LKJC", "LAJC", "tibia")
            com_accum += add_limb("R", "HU", "humerus") + add_limb("L", "HU", "humerus")
            com_accum += add_joint("REJC", "RRJC", "radius") + add_joint("LEJC", "LRJC", "radius")
            com_accum += add_limb("R", "HN", "hand") + add_limb("L", "HN", "hand")
            com_accum += add_foot("R") + add_foot("L")

            pel, v_pel = self.get_model_data("PEL", f)
            if not v_pel:
                frame_valid = False
            dd_pel, pp_pel, _, _ = self.bone_segment(pel)
            com_accum += self.weights["pelvis"] * (dd_pel + self.dist_ratios["pelvis"] * 0.925 * (pp_pel - dd_pel))
            top_l5 = dd_pel + 0.925 * (pp_pel - dd_pel)

            trx, v_trx = self.get_model_data("TRX", f)
            c7, v_c7 = self.get_marker_data("C7", f)
            if not (v_trx and v_c7):
                frame_valid = False

            dd_trx, _, R_trx, _ = self.bone_segment(trx)
            prox_c7 = c7 - (self.marker_radius * 1.05) * R_trx[:, 0]
            com_accum += self.weights["thorax"] * (prox_c7 + (1 - self.dist_ratios["thorax"]) * (top_l5 - prox_c7))

            hed, v_hed = self.get_model_data("HED", f)
            if not v_hed:
                frame_valid = False
            dd_hed, pp_hed, _, _ = self.bone_segment(hed, axis_index=0)
            com_accum += self.weights["head"] * (dd_hed + self.dist_ratios["head"] * (pp_hed - dd_hed))

            if frame_valid:
                com_results.append([com_accum[0], com_accum[1], com_accum[2], 1.0])
            else:
                com_results.append([0.0, 0.0, 0.0, 0.0])

        return np.asarray(com_results)


class ViconCoMCalculator(BaseCoMCalculator):
    def process_sequence(self, step=1, ret_vicon_ref_coms=False):
        c7_sum = np.zeros(3)
        c7_count = 0
        for f in tqdm(range(self.num_frames), desc="C7 Global", leave=False):
            trx, v_trx = self.get_model_data("TRX", f)
            c7, v_c7 = self.get_marker_data("C7", f)
            if v_trx and v_c7:
                _, _, R_trx, _ = self.bone_segment(trx)
                c7_sum += c7 - (self.marker_radius * 1.05) * R_trx[:, 0]
                c7_count += 1
        c7_global = c7_sum / c7_count if c7_count > 0 else np.zeros(3)

        c7_loc_sum = np.zeros(3)
        l5_loc_sum = np.zeros(3)
        count = 0
        axis_flip = np.diag([1, -1, -1])

        for f in tqdm(range(self.num_frames), desc="Loc Calc", leave=False):
            trx, v_trx = self.get_model_data("TRX", f)
            pel, v_pel = self.get_model_data("PEL", f)
            if v_trx and v_pel:
                dd_trx, _, R_trx, _ = self.bone_segment(trx)
                dd_pel, pp_pel, _, _ = self.bone_segment(pel)
                top_l5 = dd_pel + 0.925 * (pp_pel - dd_pel)
                c7_loc_sum += (c7_global - dd_trx) @ R_trx @ axis_flip
                l5_loc_sum += (top_l5 - dd_trx) @ R_trx @ axis_flip
                count += 1

        c7_in = c7_loc_sum / count if count > 0 else np.zeros(3)
        l5_in = l5_loc_sum / count if count > 0 else np.zeros(3)
        t_len = np.linalg.norm(l5_in - c7_in)
        t_norm = (c7_in + (l5_in - c7_in) * 0.63) / t_len if t_len > 0 else np.zeros(3)

        com_results = []
        vicon_ref_coms = []

        for f in tqdm(range(0, self.num_frames, step), desc="Vicon Rep Calc", leave=False):
            com_accum = np.zeros(3)
            frame_valid = True

            def add_limb(side, seg, w_key):
                nonlocal frame_valid
                data, valid = self.get_model_data(side + seg, f)
                if not valid:
                    frame_valid = False
                dd, pp, _, _ = self.bone_segment(data)
                return self.weights[w_key] * (dd + self.dist_ratios[w_key] * (pp - dd))

            def add_joint(p_name, d_name, w_key):
                nonlocal frame_valid
                prox, v1 = self.get_model_data(p_name, f)
                dist, v2 = self.get_model_data(d_name, f)
                if not (v1 and v2):
                    frame_valid = False
                return self.weights[w_key] * (dist[:3] + self.dist_ratios[w_key] * (prox[:3] - dist[:3]))

            def add_foot(side):
                nonlocal frame_valid
                seg, v1 = self.get_model_data(side + "FO", f)
                toe, v2 = self.get_marker_data(side + "TOE", f)
                heel, v3 = self.get_marker_data(side + "HEE", f)
                ajc, v4 = self.get_model_data(side + "AJC", f)
                if not (v1 and v2 and v3 and v4):
                    frame_valid = False
                _, _, R, _ = self.bone_segment(seg)
                scale = np.linalg.norm(heel - toe) - self.marker_radius
                return self.weights["foot"] * (ajc[:3] - self.dist_ratios["foot"] * scale * R[:, 2])

            com_accum += add_limb("R", "FE", "femur") + add_limb("L", "FE", "femur")
            com_accum += add_joint("RKJC", "RAJC", "tibia") + add_joint("LKJC", "LAJC", "tibia")
            com_accum += add_limb("R", "HU", "humerus") + add_limb("L", "HU", "humerus")
            com_accum += add_joint("REJC", "RRJC", "radius") + add_joint("LEJC", "LRJC", "radius")
            com_accum += add_limb("R", "HN", "hand") + add_limb("L", "HN", "hand")
            com_accum += add_foot("R") + add_foot("L")

            pel, v_pel = self.get_model_data("PEL", f)
            if not v_pel:
                frame_valid = False
            dd_pel, pp_pel, _, _ = self.bone_segment(pel)
            com_accum += self.weights["pelvis"] * (dd_pel + self.dist_ratios["pelvis"] * 0.925 * (pp_pel - dd_pel))

            trx, v_trx = self.get_model_data("TRX", f)
            if not v_trx:
                frame_valid = False
            dd_trx, _, R_trx, _ = self.bone_segment(trx)
            com_accum += self.weights["thorax"] * ((t_norm * t_len) @ (R_trx @ axis_flip).T + dd_trx)

            hed, v_hed = self.get_model_data("HED", f)
            if not v_hed:
                frame_valid = False
            dd_hed, pp_hed, _, _ = self.bone_segment(hed, axis_index=0)
            com_accum += self.weights["head"] * (dd_hed + self.dist_ratios["head"] * (pp_hed - dd_hed))

            if frame_valid:
                com_results.append([com_accum[0], com_accum[1], com_accum[2], 1.0])
            else:
                com_results.append([0.0, 0.0, 0.0, 0.0])

            if ret_vicon_ref_coms and "CentreOfMass" in self.model_outputs:
                v_data, v_valid = self.get_model_data("CentreOfMass", f)
                vicon_ref_coms.append([v_data[0], v_data[1], v_data[2], 1.0 if v_valid else 0.0])

        if ret_vicon_ref_coms:
            return np.asarray(com_results), np.asarray(vicon_ref_coms)
        return np.asarray(com_results)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def compute_stats(pred, gt):
    if pred is None or gt is None:
        return None

    n = min(len(pred), len(gt))
    pred = np.asarray(pred[:n])
    gt = np.asarray(gt[:n])

    pred_xyz = pred[:, :3]
    gt_xyz = gt[:, :3]
    pred_valid = pred[:, 3] > 0.5 if pred.shape[1] >= 4 else np.ones(n, dtype=bool)
    gt_valid = gt[:, 3] > 0.5 if gt.shape[1] >= 4 else np.ones(n, dtype=bool)
    valid_mask = pred_valid & gt_valid

    if not np.any(valid_mask):
        return {
            "num_frames_compared": int(n),
            "num_valid_pairs": 0,
            "mean_euclidean_mm": None,
            "std_euclidean_mm": None,
            "max_euclidean_mm": None,
        }

    diff = pred_xyz[valid_mask] - gt_xyz[valid_mask]
    d = np.linalg.norm(diff, axis=1)
    return {
        "num_frames_compared": int(n),
        "num_valid_pairs": int(valid_mask.sum()),
        "mean_euclidean_mm": float(np.mean(d)),
        "std_euclidean_mm": float(np.std(d)),
        "max_euclidean_mm": float(np.max(d)),
    }


def align_to_min_length(*arrays):
    valid_arrays = [a for a in arrays if a is not None]
    if not valid_arrays:
        return arrays, 0
    n = min(len(a) for a in valid_arrays)
    out = []
    for a in arrays:
        out.append(None if a is None else a[:n])
    return out, n


def subject_filter_pass(name, allowed):
    if not allowed:
        return True
    return name in allowed


def om_filter_pass(name, allowed):
    return takes.selected(name, takes.parse_selection(allowed))


def main():
    parser = argparse.ArgumentParser(description="Compute OM instantaneous CoM and compare to GT/model CoM.npy")
    parser.add_argument("--model-root", default=VICON_FULL)
    parser.add_argument("--gt-root", default=COMPLETE)
    parser.add_argument("--save-dir", default=COM_COMPARISON)
    parser.add_argument(
        "--inst-save-root",
        default=None,
        help="Optional root directory to save CoM_inst.npy and CoM_floor_inst.npy as <root>/<subject>/<OM>/...",
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject names, e.g. Abhinav Keaton")
    parser.add_argument("--oms", nargs="*", default=None, help="Optional OM names, e.g. OM1 OM2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    subject_dirs = [p for p in glob(os.path.join(args.model_root, "*")) if os.path.isdir(p)]
    subject_dirs = sorted(subject_dirs, key=natural_sort_key)

    overall_summary = {}

    for subject_dir in subject_dirs:
        subject = os.path.basename(subject_dir)
        if not subject_filter_pass(subject, args.subjects):
            continue

        print(f"\nProcessing {subject}...")
        save_subject_dir = os.path.join(args.save_dir, subject)
        save_data_dir = os.path.join(save_subject_dir, "data")
        os.makedirs(save_data_dir, exist_ok=True)

        om_files = sorted(glob(os.path.join(subject_dir, "*.npy")),
                          key=lambda f: takes.sort_key(os.path.splitext(os.path.basename(f))[0]))
        subject_summary = {}

        for om_file in om_files:
            om_name = os.path.splitext(os.path.basename(om_file))[0]
            if not om_filter_pass(om_name, args.oms):
                continue


            om_dir = os.path.join(save_subject_dir, om_name)
            os.makedirs(om_dir, exist_ok=True)
            save_path = os.path.join(save_data_dir, f"{om_name}_com_data.npy")
            gt_path = os.path.join(args.gt_root, subject, om_name, "CoM.npy")

            if args.skip_existing and os.path.exists(save_path):
                print(f"  {om_name}: skipping (exists)")
                continue

            print(f"  {om_name}: calculating...")
            inst_calc = InstCoMCalculator(om_file)
            inst_coms = inst_calc.process_sequence(step=args.step)

            model_vicon_coms = None
            if "CentreOfMass" in inst_calc.model_outputs:
                model_com = np.asarray(inst_calc.model_outputs["CentreOfMass"]["data"])
                model_valid = np.asarray(inst_calc.model_outputs["CentreOfMass"].get("valid", np.ones(len(model_com))))
                model_vicon_coms = np.zeros((len(model_com), 4))
                model_vicon_coms[:, :3] = model_com[:, :3]
                model_vicon_coms[:, 3] = (model_valid > 0.5).astype(float)
                model_vicon_coms = model_vicon_coms[::args.step]

            gt_coms = None
            if os.path.exists(gt_path):
                gt_coms = np.load(gt_path)
                if gt_coms.ndim == 1:
                    gt_coms = gt_coms.reshape(-1, 4)
            else:
                print(f"    GT missing: {gt_path}")

            (inst_aligned, model_vicon_aligned, gt_aligned), aligned_n = align_to_min_length(
                inst_coms, model_vicon_coms, gt_coms
            )

            stats = {
                "aligned_frames": int(aligned_n),
                "inst_vs_gt": compute_stats(inst_aligned, gt_aligned),
                "model_vicon_vs_gt": compute_stats(model_vicon_aligned, gt_aligned),
                "inst_vs_model_vicon": compute_stats(inst_aligned, model_vicon_aligned),
            }

            np.save(
                save_path,
                {
                    "subject": subject,
                    "om_name": om_name,
                    "source_model_file": om_file,
                    "source_gt_file": gt_path if os.path.exists(gt_path) else None,
                    "inst_coms": inst_coms,
                    "model_vicon_coms": model_vicon_coms,
                    "gt_coms": gt_coms,
                    "stats": stats,
                },
            )
            
            # Save just the inst. CoM
            np.save(
                os.path.join(om_dir, f"CoM_inst.npy"),
                inst_coms,
            )
            inst_floor = np.asarray(inst_coms).copy()
            if inst_floor.shape[1] >= 3:
                inst_floor[:, 2] = 0.0
            np.save(
                os.path.join(om_dir, "CoM_floor_inst.npy"),
                inst_floor,
            )

            # Optionally write directly to Complete/<subject>/<OM>/...
            if args.inst_save_root:
                inst_out_dir = os.path.join(args.inst_save_root, subject, om_name)
                os.makedirs(inst_out_dir, exist_ok=True)
                np.save(os.path.join(inst_out_dir, "CoM_inst.npy"), inst_coms)
                np.save(os.path.join(inst_out_dir, "CoM_floor_inst.npy"), inst_floor)

            subject_summary[om_name] = stats

            if stats["inst_vs_gt"] and stats["inst_vs_gt"]["mean_euclidean_mm"] is not None:
                i_mean = stats["inst_vs_gt"]["mean_euclidean_mm"]
                mv_mean = stats["model_vicon_vs_gt"]["mean_euclidean_mm"]
                print(f"    mean err vs GT (mm): inst={i_mean:.3f}, model-vicon={mv_mean:.3f}")

        overall_summary[subject] = subject_summary

    summary_path = os.path.join(args.save_dir, "summary.npy")
    np.save(summary_path, overall_summary)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
