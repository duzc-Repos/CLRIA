import numpy as np
import pandas as pd

import os

class LRdatabase(object):
    """ Extract LR regional expression and thier relationship from gene expression and LR database

    Parameters
    ----------
    lrdb: None, str or pandas.DataFrame
        if None, use built-in database which were based on CellChat and NeuroChat.
        else input your own curated database with at least 5 columns in order: (tsv format)
            "interaction_id", index column, 
            "lignad symbol": genes related to ligand, seperated by "_" if multiple gene to a lignad
            "receptor symbol", genes related to ligand, seperated by "_" if multiple gene to a lignad
            "lr_type", see lr_type for example
            "is_neurontransmitter", bool variable, if this ligand receptor interaction related to neurontransmitter.
    lr_type: list, choose from "Secreted", "ECM-Receptor", "Cell-Cell Contact", "Non-protein Signaling" or their combination.
    level: str, choose from "lr" or "gene". 
        "lr": extract ligand and receptor expression by geometric mean;
        "gene": means extract the gene expression related to ligand and receptors
    
    Attributes
    ----------
    L : (n_sender, n_ligand) pandas.DataFrame
        The ligand expression matrix.
    R : (n_receiver, n_receptor) pandas.DataFrame
        The receptor expression matrix.
    TL : (n_lr, n_ligand) pandas.DataFrame
        The generated ligand coupling matrix.
    TR : (n_lr, n_receptor) pandas.DataFrame
        The generated ligand coupling matrix.    
    """
    def __init__(self, lrdb=None, lr_type=["Secreted", "Non-protein Signaling"], level="lr"):
        if lrdb is None:
            curr_dir = os.path.dirname(__file__)
            lrdb1 = pd.read_csv(os.path.join(curr_dir, "./LRdatabase/lrdb_cell.tsv"), sep="\t", index_col=0)
            lrdb1["is_neurontransmitter"] = False
            lrdb2 = pd.read_csv(os.path.join(curr_dir, "./LRdatabase/lrdb_neuron.tsv"), sep="\t", index_col=0)
            lrdb2["is_neurontransmitter"] = True
            lrdb = pd.concat([lrdb1, lrdb2])
        elif isinstance(lrdb, str):
            lrdb = pd.read_csv(lrdb)
        else:
            pass
        assert lrdb.shape[1]>=5, "at least 5 columns for lrdb"

        self.lrdb = lrdb.loc[lrdb.iloc[:, 2].isin(lr_type), :]
        self.level = level
    
    def extract_lr_expression(self, expr, level=None, method="geo"):
        """Extract regional expression matrix of ligand and receptor.

        Parameters
        ----------
        expr : str or (n_region, n_genes) pandas.DateFrame
            The regional gene expression matrix
        level : None or str, optional
            Construct the coupling matrix at "gene" level or "lr" level, by default None
        method : str, optional
            The method to get ligand and receptor expression from genes, by default "geo"
        """
        if level is not None:
            self.level = level

        ## 00. load expression data
        expr = self._load_expr_data(expr)
        expr_colsum = expr.sum(axis=0)
        expr = expr.loc[:, expr_colsum>0]

        if level is not None:
            self.level = level

        if self.level == "lr":
            L, R = self._extract_expr_level_lr(expr, method=method)
        elif self.level == "gene":
            L, R = self._extract_expr_level_gene(expr)
        else:
            pass
        self.L = L
        self.R = R

    def generate_coupling_matrix(self, L=None, R=None, level=None):
        if level is not None:
            self.level = level
        if L is not None:
            self.L = L
        if R is not None:
            self.R = R

        TL, TR = {}, {}
        if self.level == "lr":
            for lr_id in self.lrdb.index:
                l_id, r_id = lr_id.split("|")
                if l_id in self.L.columns and r_id in self.R.columns:
                    TL[lr_id] = {l_id:1}
                    TR[lr_id] = {r_id:1}
    
        elif self.level == "gene":
            for lr_id in self.lrdb.index:
                l_id, r_id = lr_id.split("|")
                l_sym, r_sym = l_id.split("_"), r_id.split("_")
                l_isin = [ g in self.L.columns for g in l_sym ]
                r_isin = [ g in self.R.columns for g in r_sym ]
                if np.alltrue(l_isin) and np.alltrue(r_isin):
                    TL[lr_id] = { g:1 for g in l_sym}
                    TR[lr_id] = { r:1 for r in r_sym}
        else:
            pass

        ## fill nan with 0
        TL = pd.DataFrame(TL).T
        TL[np.isnan(TL)] = 0
        TR = pd.DataFrame(TR).T
        TR[np.isnan(TR)] = 0
        self.TL = TL
        self.TR = TR

    def _extract_expr_level_lr(self, expr, method):
        L, R = {}, {}
        for l_sym, r_sym in self.lrdb.values[:, :2]:
            l_syms, r_syms = l_sym.split("_"), r_sym.split("_")
            l_isin = [ l in expr.columns for l in l_syms ]
            r_isin = [ r in expr.columns for r in r_syms ]
            if (not np.alltrue(l_isin)) or (not np.alltrue(r_isin)):
                continue
            
            ## geometric mean
            if method == "geo":
                if l_sym not in L:
                    L[l_sym] = np.power( np.prod(expr.loc[:, l_syms].values, axis=1), 1/len(l_syms) )
                if r_sym not in R:
                    R[r_sym] = np.power( np.prod(expr.loc[:, r_syms].values, axis=1), 1/len(r_syms) )
            ## mean
            elif method == "mean":
                if l_sym not in L:
                    L[l_sym] = np.mean(expr.loc[:, l_syms].values, axis=1)
                if r_sym not in R:
                    R[r_sym] = np.mean(expr.loc[:, r_syms].values, axis=1)
            ## min
            elif method == "min":
                if l_sym not in L:
                    L[l_sym] = np.min(expr.loc[:, l_syms].values, axis=1)
                if r_sym not in R:
                    R[r_sym] = np.min(expr.loc[:, r_syms].values, axis=1)
            else:
                pass

        L = pd.DataFrame(L)
        L.index = expr.index
        R = pd.DataFrame(R)
        R.index = expr.index
        return L, R

    def _extract_expr_level_gene(self, expr):
        L, R = {}, {}
        for l_sym, r_sym in self.lrdb.values[:, :2]:
            l_syms, r_syms = l_sym.split("_"), r_sym.split("_")
            l_isin = [ l in expr.columns for l in l_syms ]
            r_isin = [ r in expr.columns for r in r_syms ]
            if (not np.alltrue(l_isin)) or (not np.alltrue(r_isin)):
                continue

            for g in l_syms:
                if g not in L:
                    L[g] = expr.loc[:, g].values
            for g in r_syms:
                if g not in R:
                    R[g] = expr.loc[:, g].values


        L = pd.DataFrame(L)
        L.index = expr.index
        R = pd.DataFrame(R)
        R.index = expr.index
        return L, R
    
    def _load_expr_data(self, expr):
        if isinstance(expr, str):
            if expr.endswith("parq"):
                return pd.read_parquet(expr)
            else:
                return pd.read_csv(expr)
        else:
            return expr


if __name__ == "__main__":
    import sys

    ## test1
    filename = sys.argv[1]
    level = "lr"
    lrdb_obj = LRdatabase(level=level)
    lrdb_obj.extract_lr_expression(filename)
    lrdb_obj.generate_coupling_matrix()
    print(lrdb_obj.L.shape, lrdb_obj.R.shape, lrdb_obj.TL.shape, lrdb_obj.TR.shape)

    ## test2
    filename = pd.read_parquet(sys.argv[1])
    level = "lr"
    lrdb_obj = LRdatabase(level=level)
    lrdb_obj.extract_lr_expression(filename)
    lrdb_obj.generate_coupling_matrix()
    print(lrdb_obj.L.shape, lrdb_obj.R.shape, lrdb_obj.TL.shape, lrdb_obj.TR.shape)
