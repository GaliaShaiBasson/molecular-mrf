import os
import re
import numpy as np
import scipy.io as sio

import pydicom as dcm


def _dicom_number(fn: str) -> int:
    """
    Extract XX from MRImXX.dcm.
    """
    m = re.match(r"MRIm(\d+)\.dcm$", os.path.basename(fn))
    if not m:
        raise ValueError(f"Unexpected filename format: {fn}")
    return int(m.group(1))


def load_dicom_image(path: str) -> np.ndarray:
    ds = dcm.dcmread(path)
    arr = ds.pixel_array.astype(np.float64)
    return arr


def preprocess_mt52(
    mt_folder: str,
    t1_dcm_path: str,
    t2_dcm_path: str,
    out_acq_mat: str,
    out_t12_mat: str,
    eps: float = 1e-8,
):
    # ---- Load and sort MT DICOMs ----
    fns = [fn for fn in os.listdir(mt_folder) if fn.lower().endswith(".dcm")]
    fns = sorted(fns, key=_dicom_number)

    # Expect MRIm02.dcm to MRIm32.dcm

    # MRIm02 is M0
    m0_path = os.path.join(mt_folder, fns[0])
    m0 = load_dicom_image(m0_path)

    # Remaining 30 are measurements
    meas_fns = fns[1:]  # MRIm03..MRIm32
    if len(meas_fns) != 30:
        raise ValueError(f"Expected 30 measurement images after M0, got {len(meas_fns)}")

    meas = []
    for fn in meas_fns:
        img = load_dicom_image(os.path.join(mt_folder, fn))
        meas.append(img)

    meas = np.stack(meas, axis=0)  # (30, H, W)

    # Normalize by M0 (per voxel)
    data = meas / (m0[None, :, :] + eps)  # (30, H, W)

    # Save acquired data mat
    os.makedirs(os.path.dirname(out_acq_mat), exist_ok=True)
    sio.savemat(out_acq_mat, {"data": data.astype(np.float32)})
    print(f"Saved: {out_acq_mat}")
    print("acquired_data_mt52 shape:", data.shape)

    # ---- Load T1/T2 maps (unscaled) and convert to ms using RescaleSlope ----
    ds_t1 = dcm.dcmread(t1_dcm_path)
    t1_ms = ds_t1.pixel_array.astype(np.float64) * float(ds_t1.RescaleSlope)

    ds_t2 = dcm.dcmread(t2_dcm_path)
    t2_ms = ds_t2.pixel_array.astype(np.float64) * float(ds_t2.RescaleSlope)

    sio.savemat(out_t12_mat, {"t1": t1_ms.astype(np.float32), "t2": t2_ms.astype(np.float32)})
    print(f"Saved: {out_t12_mat}")
    print("t1 shape:", t1_ms.shape, "t2 shape:", t2_ms.shape)


def main():
    base = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base, "data")

    mt_folder = os.path.join(data_dir, "mt_52")
    t1_dcm_path = os.path.join(data_dir, "t1map_unscaled.dcm")
    t2_dcm_path = os.path.join(data_dir, "t2map_unscaled.dcm")

    out_acq_mat = os.path.join(data_dir, "acquired_data_mt52.mat")
    out_t12_mat = os.path.join(data_dir, "t1_t2_maps_aging.mat")

    preprocess_mt52(
        mt_folder=mt_folder,
        t1_dcm_path=t1_dcm_path,
        t2_dcm_path=t2_dcm_path,
        out_acq_mat=out_acq_mat,
        out_t12_mat=out_t12_mat,
    )


if __name__ == "__main__":
    main()