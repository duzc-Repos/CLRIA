import numpy as np
import pandas as pd
from netneurotools import freesurfer, stats

def gen_permM(surf_files, info, hemisphere="L", n_perm=100):
    ## cortex
    coords, hemi = freesurfer.find_parcel_centroids(lhannot=surf_files[0], rhannot=surf_files[1], drop=["Unknown"])
    spins = stats.gen_spinsamples(coords=coords[hemi==0, :], hemiid=hemi[hemi==0], n_rotate=n_perm)

    ## subcortex
    info_left = info.loc[info["hemisphere"]==hemisphere, :].reset_index(drop=True)
    idx_for_permutation = info_left.loc[info_left["structure"]=="subcortex", :].index.values.copy()
    rnds = []
    for _ in range(n_perm):
        np.random.shuffle(idx_for_permutation)
        rnds.append(idx_for_permutation.copy())
    rnds = np.array(rnds).T
    return np.concatenate([spins, rnds], axis=0)  # N_region * n_perm