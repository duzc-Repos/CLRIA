import numpy as np
import pandas as pd
import scipy as sp
import tensorly as tl
import nibabel as nib
from statsmodels.stats import multitest

import os

def fdr_correction_ttest(Pmtx):
    idx = np.triu(np.ones_like(Pmtx), 1).astype(bool)
    p_flatten = multitest.fdrcorrection(Pmtx[idx], alpha=0.05, method="indep", is_sorted=False)
    return sp.spatial.distance.squareform(p_flatten[1])

def SC_L_decomposition(W):
    """
    ref: Decoupling of brain function from structure reveals regional behavioral specialization in humans, NC, 2019
         Maria Giulia Preti & Dimitri Van De Ville
    """
    
    degree = W.sum(axis=1)
    tmp = np.diag(degree**(-0.5))
    Wsymm = tmp.dot(W).dot(tmp)
    Wnew = Wsymm.copy()

    L = np.identity(W.shape[0]) - Wnew

    lambdaL, U = np.linalg.eigh(L)
    IndL = np.argsort(lambdaL)
    lambdaL = lambdaL[IndL]
    U = U[:, IndL]
    return lambdaL, U

class TCA(object):
    def __init__(self, A, B, C, lrdb_obj=None):
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()

        tmp = [self.A.sum(axis=0, keepdims=True), self.B.sum(axis=0, keepdims=True)]
        self.A /= tmp[0]
        self.B /= tmp[1]
        self.C *= (tmp[0] * tmp[1])

        self.N1 = self.A.shape[0]
        self.N2 = self.B.shape[0]
        self.n_lr = self.C.shape[0]
        self.rank = self.A.shape[1]

        if lrdb_obj is not None:
            self.L = lrdb_obj.L.copy()
            self.R = lrdb_obj.R.copy()
            self.lr_anno = lrdb_obj.lrdb.loc[lrdb_obj.TL.index, :]

            
            self.A_ = pd.DataFrame(self.A).copy()
            self.A_.index = self.L.index
            self.A_.columns = [ f"Pattern_{i+1:02d}" for i in range(self.rank) ]

            self.B_ = pd.DataFrame(self.B).copy()
            self.B_.index = self.R.index
            self.B_.columns = [ f"Pattern_{i+1:02d}" for i in range(self.rank) ]
            
            self.C_ = pd.DataFrame(self.C).copy()
            self.C_.index = self.lr_anno.index
            self.C_.columns = [ f"Pattern_{i+1:02d}" for i in range(self.rank) ]

            self.lr_anno = lrdb_obj.lrdb.loc[self.C_.index, :]
        
        self.t_stat = None
        self.t_fdr = None

    def calc_T_statistics_matrix(self):
        t_stat = np.zeros(shape=(self.N1, self.N2))
        t_pval = np.zeros(shape=(self.N1, self.N2))
        for i in range(self.N1-1):
            for j in range(i+1, self.N2):
                vec_i2j = self.C.dot( (self.A[[i], :] * self.B[[j], :]).T )
                vec_j2i = self.C.dot( (self.A[[j], :] * self.B[[i], :]).T )

                t,  p1 = sp.stats.ttest_1samp(vec_i2j-vec_j2i, 0)

                t_stat[i, j], t_pval[i, j] = t, p1
                t_stat[j, i], t_pval[j, i] = -t, p1

        t_fdr = fdr_correction_ttest(t_pval)
        self.t_stat = t_stat
        self.t_fdr = t_fdr

    def get_pathway_network(self, pathway_name, row_norm_Ce=True, return_LR_name=False):
        idx = self.lr_anno["pathway"] == pathway_name
        select_Ce = self.C_.loc[idx.values, :]
        select_name = select_Ce.index.values
        select_Ce = select_Ce.values
        if row_norm_Ce:
            select_Ce = (select_Ce - select_Ce.min(axis=1, keepdims=True)) / (select_Ce.max(axis=1, keepdims=True) - select_Ce.min(axis=1, keepdims=True))

        pathway_net = []
        for i in range(select_Ce.shape[0]):
            pathway_net.append( np.dot(self.A * select_Ce[[i], :], self.B.T) )
        pathway_net = np.array(pathway_net)

        if return_LR_name:
            return pathway_net, select_name
        else:
            return pathway_net
        pass

    def get_LR_network(self, idx, row_norm_Ce=True, return_LR_name=False):
        """Reconstruct the network of the given LR pairs

        Parameters
        ----------
        idx : list
            The index in Ce, a int or str list is required.
        return_LR_name : bool, optional
        
        Return
        ------
        lr_network : numpy.ndarray
        lr_name : numpy.ndarray, optional

        """
        ## type check
        if np.alltrue([ isinstance(i, str) for i in idx]):
            select_Ce = self.C_.loc[idx, :]
        elif np.alltrue([ isinstance(i, int) for i in idx]):
            select_Ce = self.C_.iloc[idx, :]
        else:
            raise ValueError("The data types in the list must all be str or int.")
        select_name = select_Ce.index.values
        select_Ce = select_Ce.values
        if row_norm_Ce:
            select_Ce = (select_Ce - select_Ce.min(axis=1, keepdims=True)) / (select_Ce.max(axis=1, keepdims=True) - select_Ce.min(axis=1, keepdims=True))

        
        lr_network = []
        for i in range(select_Ce.shape[0]):
            lr_network.append( np.dot(self.A * select_Ce[[i], :], self.B.T) )
        lr_network = np.array(lr_network)

        if return_LR_name:
            return lr_network, select_name
        else:
            return lr_network

        pass

    def analyze_trans_hierarchical_signaling_cortex(self, hierarchy, order="ascending",
                                                    drop_nonsignificant = False, drop_negative = True):
        """Trans-hierarchical aysmmetric signaling

        Parameters
        ----------
        hierarchy : None or numpy.ndarray
            A float array recording the hierarchical values inffered from Functional gradient, SAaxis, myelin, etc.
            If nan values in array, the corresponding region would be ignored.
            If None, Using default order of factor matrix A, by default None.
        order : str
            Sort hierarchy in "ascending" or "descending"
        drop_nonsignificant : bool, optional
            If True, the T-statistic matrix with FDR>0.05 would set to zero, by default False.
        drop_negative : bool, optional
            If True, the T-statistic matrix with values < 0 would set to zero, by default True.
        
        Return
        ------
        low2high : numpy.ndarray
            The T-statistics of lower-to-higher hierarchical regions.
        high2low : numpy.ndarray
            The T-statistics of lower-to-higher hierarchical regions.
        pval : (KS, P_KS) tuple
            The P-values of KS test.
        """
        assert self.N1 == self.N2, f"Non-square matrix doesn't support trans-hierarchial signaling analysis! ({self.N1}, {self.N2})"            
        assert order in {"ascending", "descending"}, "Error values, only 'ascending' or 'descending' are allowed."
        if self.t_stat is None:
            self.calc_T_statistics_matrix()
        
        t_stat = self.t_stat.copy()
        t_fdr = self.t_stat.copy()
        if drop_nonsignificant:
            t_stat[t_fdr>=0.05] = 0
        if drop_negative:
            t_stat[t_stat<0] = 0
        
        if hierarchy is None:
            use_region = np.array([True] * self.N1)
            use_hierarchy = np.arange(self.N1)
        else:
            use_region = ~np.isnan(hierarchy)
            use_hierarchy = hierarchy[use_region]
        use_hierarchy_idx = np.argsort(use_hierarchy)
        if order == "descending":
            use_hierarchy_idx = use_hierarchy_idx[::-1]

        t_stat = t_stat[use_region, :][:, use_region]
        t_stat = t_stat[use_hierarchy_idx, :][:, use_hierarchy_idx]
        upper_idx = np.triu(np.ones_like(t_stat) ,k=1).astype(bool)
        low2high = t_stat[upper_idx]
        high2low = t_stat[upper_idx.T]
        
        tmp_ks = sp.stats.ks_2samp(low2high, high2low, method="exact", alternative="less")  ## Null: F(x) >= G(x) Alter: F(x) < G(x)
        ks, p_ks = tmp_ks.statistic, tmp_ks.pvalue
        return low2high, high2low, (ks, p_ks)

    def analyze_trans_hierarchical_signaling_cortex_spin(self, hierarchy, order="ascending", perm_idx=None,
                                                         drop_nonsignificant = False, drop_negative = True):
        """Trans-hierarchical aysmmetric signaling

        Parameters
        ----------
        hierarchy : None or numpy.ndarray
            A float array recording the hierarchical values inffered from Functional gradient, SAaxis, myelin, etc.
            If nan values in array, the corresponding region would be ignored.
            If None, Using default order of factor matrix A, by default None.
        order : str, optional
            Sort hierarchy in "ascending" or "descending"
        perm_idx : None or (n_region, n_perm) numpy.ndarray
            The permutation index generated by random or spatial autocorrelation model.
        drop_nonsignificant : bool, optional
            If True, the T-statistic matrix with FDR>0.05 would set to zero, by default False.
        drop_negative : bool, optional
            If True, the T-statistic matrix with values < 0 would set to zero, by default True.
        
        Return
        ------
        diff_empi : float
        diff_perm : numpy.ndarray
        p_spin : float
        """
        assert self.N1 == self.N2, f"Non-square matrix doesn't support trans-hierarchial signaling analysis! ({self.N1}, {self.N2})"            
        assert order in {"ascending", "descending"}, "Error values, only 'ascending' or 'descending' are allowed."
        if self.t_stat is None:
            self.calc_T_statistics_matrix()
        
        t_stat = self.t_stat.copy()
        t_fdr = self.t_stat.copy()
        if drop_nonsignificant:
            t_stat[t_fdr>=0.05] = 0
        if drop_negative:
            t_stat[t_stat<0] = 0
        
        if hierarchy is None:
            use_region = np.array([True] * self.N1)
            use_hierarchy = np.arange(self.N1)
        else:
            use_region = ~np.isnan(hierarchy)
            use_hierarchy = hierarchy[use_region]
        use_hierarchy_idx = np.argsort(use_hierarchy)
        if order == "descending":
            use_hierarchy_idx = use_hierarchy_idx[::-1]

        t_stat = t_stat[use_region, :][:, use_region]
        t_stat = t_stat[use_hierarchy_idx, :][:, use_hierarchy_idx]
        upper_idx = np.triu(np.ones_like(t_stat) ,k=1).astype(bool)
        low2high = t_stat[upper_idx]
        high2low = t_stat[upper_idx.T]
        
        median_empi = np.median(low2high) - np.median(high2low)

        median_perm = []
        perm_idx = perm_idx[use_region, :]
        for p_idx in perm_idx.T:
            idx = np.argsort(use_hierarchy[p_idx])
            if order == "descending":
                idx = idx[::-1]
            t_perm = t_stat[idx, :][:, idx].copy()
            l2h_perm, h2l_perm = t_perm[upper_idx], t_perm[upper_idx.T]
            median_perm.append( np.median(l2h_perm)-np.median(h2l_perm) )
        median_perm = np.array(median_perm)
        p_spin = ((median_perm >= median_empi).sum()+1) / (perm_idx.shape[1]+1)
        return median_empi, median_perm, p_spin
    
    def analyze_trans_hierarchical_signaling_cortex_and_subcortex(self, hierarchy,
                                                    drop_nonsignificant = False, drop_negative = True):
        """Trans-hierarchical aysmmetric signaling

        Parameters
        ----------
        hierarchy : (numpy.ndarray, numpy.ndarry) tuple
            Two bool index recording the subcortical and cortical regions.
        drop_nonsignificant : bool, optional
            If True, the T-statistic matrix with FDR>0.05 would set to zero, by default False.
        drop_negative : bool, optional
            If True, the T-statistic matrix with values < 0 would set to zero, by default True.
        
        Return
        ------
        low2high : numpy.ndarray
            The T-statistics of lower-to-higher hierarchical regions.
        high2low : numpy.ndarray
            The T-statistics of lower-to-higher hierarchical regions.
        pval : (KS, P_KS) tuple
            The P-values of KS test.
        """
        assert self.N1 == self.N2, f"Non-square matrix doesn't support trans-hierarchial signaling analysis! ({self.N1}, {self.N2})"            
        if self.t_stat is None:
            self.calc_T_statistics_matrix()
        
        t_stat = self.t_stat.copy()
        t_fdr = self.t_stat.copy()
        if drop_nonsignificant:
            t_stat[t_fdr>=0.05] = 0
        if drop_negative:
            t_stat[t_stat<0] = 0
        
        low2high = t_stat[hierarchy[0], :][:, hierarchy[1]].ravel()
        high2low = t_stat[hierarchy[1], :][:, hierarchy[0]].ravel()
        
        tmp_ks = sp.stats.ks_2samp(low2high, high2low, method="exact", alternative="less")  ## Null: F(x) >= G(x) Alter: F(x) < G(x)
        ks, p_ks = tmp_ks.statistic, tmp_ks.pvalue
        return low2high, high2low, (ks, p_ks)

    def graph_spectrurm_analysis(self, hamonics=None, SC=None, ):
        
        ## get hamonis in ascending order
        if hamonics is None:
            if SC is None:
                raise ValueError("It shouldn't be all None for 'harmonics' and 'SC' ")
            else:
                _, hamonics = SC_L_decomposition(SC)
        
        ## get asymmetric communication for each LR pair
        Asum0, Bsum0 = self.A.sum(axis=0, keepdims=True), self.B.sum(axis=0, keepdims=True)
        sending = (self.A * Bsum0).dot(self.C.T)
        receiving = (self.B * Asum0).dot(self.C.T)
        delta = (sending - receiving).astype(np.float128)
        mean = delta.mean(axis=0, keepdims=True)
        delta = (delta - mean)

        ## decomposition
        return (np.dot(hamonics.T, delta)**2).T
    
    def identify_dominant_LR_from_Ce(self, is_norm=True, q=0.8):
        if is_norm:
            self.Ce_norm = (self.C_.T / self.C_.sum(axis=1)).T
        else:
            self.Ce_norm = self.C_.copy()
        self.Ce_bool = self.Ce_norm >= self.Ce_norm.quantile(q=q, axis=0)
    
    def prepare_ROI_from_Ae_Be(self, atlas_nii, out_path, hemi="L", thr_Ae=None, thr_Be=None):
        if thr_Ae is None:
            thr_Ae = 1 / self.rank
        if thr_Be is None:
            thr_Be = 1 / self.rank
        
        if isinstance(atlas_nii, str):
            atlas_nii = nib.load(atlas_nii)
        elif isinstance(atlas_nii, nib.nifti1.Nifti1Image):
            atlas_nii = atlas_nii
        atlas_dat = atlas_nii.get_fdata()
        
        ## processing Ae
        Ae_norm = self.A / self.A.max(axis=1, keepdims=True)
        Ae_norm = Ae_norm >= thr_Ae
        for i in range(Ae_norm.shape[1]):
            roi_label = np.argwhere(Ae_norm[:, i]).ravel()
            if hemi == "L":
                roi_label = roi_label * 2 + 1
            elif hemi == "R":
                roi_label = roi_label * 2 + 2
            else:
                pass
            
            roi_mask = np.zeros_like(atlas_dat)
            roi_mask[np.isin(atlas_dat, roi_label)] = 1
            roi_nii = nib.Nifti1Image(roi_mask, atlas_nii.affine)
            roi_nii.to_filename( os.path.join(out_path, f"Ae_pattern{i+1:02d}.nii") )
            pass
        
        ## processing Be
        Be_norm = self.B / self.B.max(axis=1, keepdims=True)
        Be_norm = Be_norm >= thr_Be
        for i in range(Be_norm.shape[1]):
            roi_label = np.argwhere(Be_norm[:, i]).ravel()
            if hemi == "L":
                roi_label = roi_label * 2 + 1
            elif hemi == "R":
                roi_label = roi_label * 2 + 2
            else:
                pass
            
            roi_mask = np.zeros_like(atlas_dat)
            roi_mask[np.isin(atlas_dat, roi_label)] = 1
            roi_nii = nib.Nifti1Image(roi_mask, atlas_nii.affine)
            roi_nii.to_filename( os.path.join(out_path, f"Be_pattern{i+1:02d}.nii") )
            pass