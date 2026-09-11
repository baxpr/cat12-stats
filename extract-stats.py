#!/usr/bin/env python

import argparse
import numpy
import pandas
import scipy

# is_mat_struct and walk functions for matlab structures courtesy ChatGPT-5.2

def is_mat_struct(x):
    return hasattr(x, "_fieldnames") and hasattr(x, "__dict__")

def walk(obj, path=""):
    if is_mat_struct(obj):
        for name in obj._fieldnames:
            v = getattr(obj, name)
            p = f"{path}.{name}" if path else name
            yield from walk(v, p)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from walk(v, p)
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
        return

    if isinstance(obj, numpy.ndarray):
        if obj.dtype == object:
            arr = numpy.squeeze(obj)
            if arr.shape == ():
                yield from walk(arr.item(), path)
            else:
                for idx in numpy.ndindex(arr.shape):
                    yield from walk(arr[idx], f"{path}{list(idx)}")
        else:
            yield (path, obj)
        return

    yield (path, obj)
    

parser = argparse.ArgumentParser()
parser.add_argument('--tiv_txt', default='INPUTS/cat12/TIV/TIV.txt')
parser.add_argument('--cat_t1_mat', default='INPUTS/cat12/MAT/cat_t1.mat')
parser.add_argument('--catROI_t1_mat', default='INPUTS/cat12/LABEL/catROI_t1.mat')
parser.add_argument('--catROIs_t1_mat', default='INPUTS/cat12/LABEL/catROIs_t1.mat')
parser.add_argument('--out_dir', default='/OUTPUTS')
args = parser.parse_args()


tiv = pandas.read_csv(args.tiv_txt, sep='\t', header=None, names=['TIV','GM','WM','CSF','ICVfrac'])
cat_t1 = scipy.io.loadmat(args.cat_t1_mat, squeeze_me=True, struct_as_record=False)
catROI_t1 = scipy.io.loadmat(args.catROI_t1_mat, squeeze_me=True, struct_as_record=False)
catROIs_t1 = scipy.io.loadmat(args.catROIs_t1_mat, squeeze_me=True, struct_as_record=False)

for p, v in walk(catROIs_t1):
    print(p, "=", v)

