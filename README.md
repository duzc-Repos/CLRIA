# CLRIA

CLRIA: connectome-constrained ligand-receptor interaction analysis for understanding brain network communication

<img src="./img/Fig1.png" alt="Fig1" style="zoom: 45%;" />



## Installation

*   Install from PyPI: `pip install clria`

*   install from source:

    ```bash
    git clone git@github.com:duzc-Repos/CLRIA.git
    cd CLRIA
    pip install .
    ```

*   [Optional] install pytorch for GPU acceleration, see [pytorch.org](https://pytorch.org/) for detail.



## Usage

### Quick example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import clria

## 0. simulation data
n_rank = 5
n_region = 100
n_lrpair = 20
eps = 1e-100
r, A, B, C, L, R, TL, TR, M, (M1, M2) = clria.otsim.simu_data(
            n_r=n_rank, n1=n_region, n_lr=n_lrpair,             ## r, A, B, C
            TL=10, TR=15,                                       ## L, R, TL, TR
            scale_factor=1e4, epsilon=1,                        ## M, 
            is_decomposition=True, n_d=10, solver="nmf",        ## (M1, M2)
            eps=eps)

## 1. solver
epsi, tau = 0, 1
r = 5
mmot_obj = clria.ottl.MMOTNTD(L, R, M, TL, TR, epsilon=epsi, tau1=tau, tau2=tau,)
mmot_obj.fit(r=r)
(Ae, Be, Ce) = mmot_obj.factors
tca_obj = clria.ottl.TCA(Ae, Be, Ce)

## 2. plotting
annot = pd.DataFrame({ i:[f"network_{i//20+1}"] for i in range(100) }).T ## random generate annotation
clria.otpl.plot_sankey(tca_obj, annot, figure_size=(600, 700))

```



### Notebook

*   See `./tutorial/tutorial.ipynb` for an application of CLRIA to publicly available gene expression (AHBA) and diffusion MRI (HCP) data. The data used in this tutorial can be found at https://doi.org/10.5281/zenodo.13906808

*   The code to replicate the analysis of the study are available at https://doi.org/10.5281/zenodo.13906808.

## API

Here is a brief list of the functionalities provided by this repository. Additional information is provided in their docstring or by calling `help(function_name)`.

### preprocessing: otpl

*   `otpl.LRdatabse(lrdb, ...)`: 
    *   `.extract_lr_expression(self, expr, ...)`: extract ligand and receptor expression from gene expression matrix.
    *   `.generate_coupling_matrix(self, ...)`: generate two matrix $T_L, T_R$ recoding the correspondence between LR pairs and ligand or receptor.
*   `otpl.BNCM()`:
    *   `.proximal_to_distance(self, SC)`: transfrom the fiber density measure to distance measures.
    *   `.calc_navigation(self, L, D)`: calculate navigation measure based on $L$ (strength-to-length remapping of the connection weight) and $D$ (distance matrix, e.g. Euclidean distance between nodes)

### tools: ottl

*   `ottl.MMOTNTD(L, R, M, TL, TR, ...)`
    *   `.fit(self, r)`: solve LR-mediated communication patterns $A, B, C$.
*   `ottl.gen_PermM(surf_files, info, ...)`:  generate spatial-preserving permutation idx. 
*   `ottl.TCA(A, B, C)`: 
    *   `.calc_T_statistics_matrix(self)`: infer region-wise asymmetrix signaling across all LR pairs.
    *   `.analyze_trans_hierarchical_signaling_cortex(self, hierarchy, ...)`: infer trans-hierarchical asymmetrix signaling between cortical regions by KS test.
    *   `.analyze_trans_hierarchical_signaling_cortex_spin(self, hierarchy, ...)`: infer trans-hierarchical asymmetrix signaling between cortical regions by spin test.
    *   `.analyze_trans_hierarchical_signaling_cortex_and_subcortex(self, hierarchy, ...)`: infer trans-hierarchical asymmetrix signaling between cortical and subcortex regions by KS test.
    *   `.graph_spectrum_analysis(self, hamonics, SC)`: energy density analysis of asymmetrix signal of each LR pair.
    *   `.get_pathway_network(self, pathway_name, ...)`: reconstruct pathway-specific communication network from communication pattters.
    *   `.get_lr_network(self, idx, ...)`: reconstruct LR-specific communication network from communication pattters.
    *   `.identify_dominant_LR_from_Ce(self, ...)`: identify domiant LR pairs for each tensor factor.
    *   `.prepare_ROI_from_Ae_Be(self, atlas_nii, ...)`: prepare ROI files of each tensor factor for Neurosynth analysis.
*   `ottl.UOTRegression(pattern, ...)`
    *   `.fit(src, dst, ...)`: solve UOT regression between source state and target state.
    *   `.get_opt_UOT_dist(self, beta, P)`: compute UOT distance value.
    *   `.get_opt_obj_val(self, beta, P, src, dst)`: compute objective function values.
*   `ottl.OTRegression(pattern, ...)`: an balanced version of `UOTRegression`
    *   `.fit(src, dst, ...)`: solve OT regression between source state and target state.
    *   `.get_opt_OT_dist(self, beta, P)`: compute OT distance value.
    *   `.get_opt_obj_val(self, beta, P, src, dst)`: compute objective function values.

### plotting: otpl

*   `.otpl.plot_sankey(tca_obj, annot, ...)`: sankey plot of communication patterns, LR or pathway communication network.
*   `.otpl.plot_circos(tca_obj, annot, name, ...)`: circos plot of communication patterns, LR or pathway communication network.

### simmulation: otsim

*   `otsim.simu_data(n_r, n1, n2, n_lr, ...)`: simulating data using inverse optimal transport (iOT).

