#!/usr/bin/env python

import argparse
import numpy
import os
import pandas
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--cat_t1_mat', default='INPUTS/cat12/MAT/cat_t1.mat')
parser.add_argument('--catROI_t1_mat', default='INPUTS/cat12/LABEL/catROI_t1.mat')
parser.add_argument('--catROIs_t1_mat', default='INPUTS/cat12/LABEL/catROIs_t1.mat')
parser.add_argument('--out_dir', default='OUTPUTS')
args = parser.parse_args()


# is_mat_struct and walk functions for matlab structures courtesy ChatGPT-5.2
def is_mat_struct(x):
    return hasattr(x, "_fieldnames") and hasattr(x, "__dict__")

def mat_walk(obj, path=""):
    if is_mat_struct(obj):
        for name in obj._fieldnames:
            v = getattr(obj, name)
            p = f"{path}.{name}" if path else name
            yield from mat_walk(v, p)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from mat_walk(v, p)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from mat_walk(v, f"{path}[{i}]")
        return
    if isinstance(obj, numpy.ndarray):
        if obj.dtype == object:
            arr = numpy.squeeze(obj)
            if arr.shape == ():
                yield from mat_walk(arr.item(), path)
            else:
                for idx in numpy.ndindex(arr.shape):
                    yield from mat_walk(arr[idx], f"{path}{list(idx)}")
        else:
            yield (path, obj)
        return
    yield (path, obj)


# Load stats results
info = {
    'cat_t1': scipy.io.loadmat(args.cat_t1_mat, squeeze_me=True, struct_as_record=False),
    'catROI_t1': scipy.io.loadmat(args.catROI_t1_mat, squeeze_me=True, struct_as_record=False),
    'catROIs_t1': scipy.io.loadmat(args.catROIs_t1_mat, squeeze_me=True, struct_as_record=False),
    }

# Save text version of complete outputs to file
for stem in ['cat_t1', 'catROI_t1', 'catROIs_t1']:
    with open(os.path.join(args.out_dir, f'{stem}.txt'), 'w', encoding='utf-8') as f:
        for p, v in mat_walk(info[stem]):
            f.write(f'{p} = {v}\n')

# Custom csv for redcap with QC info
stats = {
    'vol_abs_CSF_cm3': info['cat_t1']['S'].subjectmeasures.vol_abs_CGW[0],
    'vol_abs_GM_cm3': info['cat_t1']['S'].subjectmeasures.vol_abs_CGW[1],
    'vol_abs_WM_cm3': info['cat_t1']['S'].subjectmeasures.vol_abs_CGW[2],
    'vol_rel_CSF': info['cat_t1']['S'].subjectmeasures.vol_rel_CGW[0],
    'vol_rel_GM': info['cat_t1']['S'].subjectmeasures.vol_rel_CGW[1],
    'vol_rel_WM': info['cat_t1']['S'].subjectmeasures.vol_rel_CGW[2],
    'vol_TIV_cm3': info['cat_t1']['S'].subjectmeasures.vol_TIV,
    'dist_thickness_mm': info['cat_t1']['S'].subjectmeasures.dist_thickness[0],
    'dist_thickness_std': info['cat_t1']['S'].subjectmeasures.dist_thickness[1],
    'res_RMS_abs_mm': info['cat_t1']['S'].qualitymeasures.res_RMS,
    'NCR_abs': info['cat_t1']['S'].qualitymeasures.NCR,
    'ICR_abs': info['cat_t1']['S'].qualitymeasures.ICR,    
    'res_RMS_rel': info['cat_t1']['S'].qualityratings.res_RMS,
    'NCR_rel': info['cat_t1']['S'].qualityratings.NCR,
    'ICR_rel': info['cat_t1']['S'].qualityratings.ICR,
    'IQR': info['cat_t1']['S'].qualityratings.IQR,
    'EC_abs': info['cat_t1']['S'].subjectmeasures.EC_abs,
    'defect_size': info['cat_t1']['S'].subjectmeasures.defect_size,
    }

# Compute percentage and grade letter ratings
# https://github.com/ChristianGaser/cat12/blob/5d87f33a5cccd9a52a4fa54cd499d77fd77b5e75/cat_main_reportcmd.m#L24
# Code conversion courtesy ChatGPT-5.2
grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E+', 'E', 'E-', 'F'];
ngrades = len(grades)
for var in ['res_RMS_rel', 'NCR_rel', 'ICR_rel', 'IQR']:
    mark = numpy.asarray(stats[var], dtype=float)
    rps = numpy.clip(105 - mark*10, 0, 100)
    rps = rps + numpy.isnan(mark) * mark
    stats[f'{var}_rps'] = rps

    if numpy.isnan(mark):
        idx_1based = ngrades
    else:
        idx_1based = int(numpy.round((mark + 2/3) * 3 - 3))
        idx_1based = max(1, idx_1based)
    idx_1based = min(ngrades, idx_1based)
    grad = grades[idx_1based - 1]
    stats[f'{var}_grade'] = grad

# Save stats to file
dstats = pandas.DataFrame([stats])
dstats = dstats.sort_index(axis=1)
dstats.to_csv(os.path.join(args.out_dir, 'stats.csv'), index=False)


# FIXME Extract ROI, ROIs stats to their own files
