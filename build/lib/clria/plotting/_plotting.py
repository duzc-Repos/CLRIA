import numpy as np
import pandas as pd
import scipy as sp

import plotly.express as px
import plotly.graph_objects as go
from pycirclize import Circos
from pycirclize.utils import ColorCycler

def plot_sankey(tca_obj, annot, loc=None, select_pattern=None, pad=8,
                title=None, save_name=None, is_show=True,
                font_size=16, figure_size=(800, 600)):
    """Sankey plot of communication patterns

    Parameters
    ----------
    tca_obj : TCA object 
        Constructed using A, B, C
    annot : pandas.DataFrame
        A regional annotation files
    loc : None or (N, 2) numpy.ndarray, optional
        The coordinate to plot the node. The number should be in[0, 1], by default None
    select_pattern : list, optional
        The index of selected patterns to plot. by default None
    pad : int, optional
        _description_, by default 8
    title : None or str, optional
        The name of the plot, by default None
    save_name : None or str, optional
        The path to save the figure, by default None
    is_show : bool, optional
        Whether to show the figure, by default True
    font_size : int, optional
        by default 16
    figure_size : tuple, optional
        by default (800, 600)
    """
    
    ## set node data
    nodeA = annot.copy()
    nodeB = annot.copy()
    nodeA["Pattern"] = "Pattern_"+pd.Series(tca_obj.A.argmax(axis=1)+1).astype(str)
    nodeB["Pattern"] = "Pattern_"+pd.Series(tca_obj.B.argmax(axis=1)+1).astype(str)

    ## set link data
    select_name = annot.columns.tolist()
    linkA = nodeA.value_counts(subset=select_name+["Pattern"], sort=False).reset_index()
    linkA.columns = select_name + ["Pattern", "Value"]
    linkB = nodeB.value_counts(subset=select_name+["Pattern"], sort=False).reset_index()
    linkB.columns = select_name + ["Pattern", "Value"]

    ## prepare data
    unique_select_name = annot[select_name[0]].unique().tolist()
    n_select_name = len(unique_select_name)
    if select_pattern is None:
        label_ = unique_select_name + [f"Pattern_{i}" for i in range(1, tca_obj.rank+1)] + unique_select_name
    else:
        label_ = unique_select_name + [f"Pattern_{i}" for i in select_pattern] + unique_select_name
    
    if loc is None:
        NODES = dict(label = label_, pad=pad)
    else:
        NODES = dict(label = label_, x = loc[:, 0], y = loc[:, 1], pad=pad)

    n_label = len(NODES["label"])
    n_add = n_label - n_select_name
    name2idx = dict( zip( NODES["label"][:-n_select_name], range(n_add) ) )
    LINKS = dict(
        source = linkA[select_name[0]].map(name2idx).values.tolist() + linkB["Pattern"].map(name2idx).values.tolist(),
        target = linkA["Pattern"].map(name2idx).values.tolist() + (linkB[select_name[0]].map(name2idx).values + n_add).tolist(),
        value  = linkA["Value"].values.tolist() + linkB["Value"].values.tolist()
    )

    ## plot
    data = go.Sankey(arrangement='freeform', node = NODES, link = LINKS)
    fig = go.Figure(data)
    if title is None:
        fig.update_layout(font_size=font_size, width=figure_size[0], height=figure_size[1] )
    else:
        fig.update_layout(title=title,
                          font_size=font_size, width=figure_size[0], height=figure_size[1])
    if save_name is not None:
        fig.write_image(save_name, format='pdf')
    if is_show:
        fig.show()
    pass


def get_lr_name_list(lr_list):
    l_names, r_names = set(), set()
    for tmp in lr_list:
        a, b = tmp.split('|', 1)
        l_names.add(a)
        r_names.add(b)
    return list(l_names), list(r_names)

def get_range(array):
    array_min, array_max = array.min(), array.max()
    array_min_int, array_max_int = int(array_min)-0.5, int(array_max)+0.5
    l = array_min_int if array_min_int <= array_min else array_min_int-0.5
    r = array_max_int if array_max_int >= array_max else array_max_int+0.5
    #if l<0 and r<0:
    #    r = 0
    #elif l>0 and r>0:
    #    l = 0
    return l, r

def plot_circos(tca_obj, annot, name, is_LR,
                strength_thr=0.0005, is_norm_expr=[True, True],
                sector_colors = ["#729FD3", "#E8C0A3", "#DAE1F2", "#A9CDB8", "#D8B8D7", "#C8C897", "#666666"],
                source_lobe = None, target_lobe = None,
                circ_lims = [50, 360], alpha_scale = 1, is_show = True, save_name = None
                ):
    ## 01. preprocessing: get network
    if is_LR:
        assert name in tca_obj.lr_anno.index, "The name of LR pair must be in the index of tca_obj.lr_anno."
        net = tca_obj.get_LR_network(idx=[name], return_LR_name=False).mean(axis=0)
        l_name, r_name = name.split("|")
        l_expr, r_expr = tca_obj.L[l_name].values, tca_obj.R[r_name].values
    else:
        assert name in tca_obj.lr_anno["pathway"].values, "The name of LR pair must be in the tca_obj.lr_anno['pathway']."
        net, lr_names = tca_obj.get_pathway_network(pathway_name=name, return_LR_name=True)
        net = net.mean(axis=0)
        l_names, r_names = get_lr_name_list(lr_names)
        l_expr, r_expr = tca_obj.L[l_names].mean(axis=1).values, tca_obj.R[r_names].mean(axis=1).values

    net /= net.max()
    if not (is_norm_expr is None):
        if is_norm_expr[0]:
            l_expr = sp.stats.zscore(l_expr)
        if is_norm_expr[1]:
            r_expr = sp.stats.zscore(r_expr)

    ## 02. preprocessing: get annotation
    Lobe = annot["Lobe"].str.split(" ", expand=True, n=1)[0].drop_duplicates().values
    lobe_reindex = np.concatenate([ annot.loc[annot["Lobe"].str.find(i)>=0].copy().reset_index().index.values for i in Lobe])
    assert len(Lobe) == len(sector_colors), "The number of lobe must match the number of given colors in 'sector colors' "

    ## 03. plot
    sectors = { i:(annot["Lobe"].str.find(i) >= 0).sum() for i in Lobe }
    sector_colors = { i:j for i, j in zip(Lobe, sector_colors) }

    circos = Circos(sectors, space=2, start=circ_lims[0], end=circ_lims[1],)
    for sector in circos.sectors:
        #sector.axis(fc="none", ls="dashdot", lw=2, ec="black", alpha=0.5)
        #sector.text(f"{sector.name} ({sector.size})", r=60, size=10)
        tmp_idx = annot["Lobe"].str.find(sector.name)>=0

        ## 01. name label
        lobe_track = sector.add_track( r_lim=(65, 72) )
        lobe_track.text(sector.name, size=12)
        lobe_track.axis(fc=sector_colors[sector.name])

        ## 03. receptor
        x, y = np.arange(sector.start, sector.end)+0.5, r_expr[tmp_idx]
        tmp_range = get_range(r_expr[tmp_idx])
        receptor_track = sector.add_track( r_lim=(74, 86) )
        receptor_track.bar(x, y, vmin=tmp_range[0], vmax=tmp_range[1], width=0.5)
        receptor_track.line(x, y, vmin=tmp_range[0], vmax=tmp_range[1], lw=1, color="gray")
        receptor_track.line(np.arange(sector.start, sector.end+1), [0]*(len(x)+1), 
                        vmin=tmp_range[0], vmax=tmp_range[1], lw=1.5, color="black")
        receptor_track.axis()
        
        ## 04. ligand expression
        x, y = np.arange(sector.start, sector.end)+0.5, l_expr[tmp_idx]
        tmp_range = get_range(l_expr[tmp_idx])
        ligand_track = sector.add_track( r_lim=(88, 100) )
        ligand_track.bar(x, y, vmin=tmp_range[0], vmax=tmp_range[1], width=0.5)
        ligand_track.line(x, y, vmin=tmp_range[0], vmax=tmp_range[1], lw=1, color="gray")
        ligand_track.line(np.arange(sector.start, sector.end+1), [0]*(len(x)+1), 
                        vmin=tmp_range[0], vmax=tmp_range[1], lw=1.5, color="black")
        ligand_track.axis()

        ## 05 add links
        tmp_net = net[tmp_idx, :]
        for i in range(tmp_net.shape[0]):
            if source_lobe is None or sector.name in source_lobe:
                param1 = (sector.name, i, i+1)
            else:
                continue          
            for j in range(tmp_net.shape[1]):
                j_name = annot["Lobe"].iloc[j].split(" ")[0]
                if target_lobe is None or j_name in target_lobe:
                    param2 = (j_name, lobe_reindex[j], lobe_reindex[j]+1)
                else:
                    continue
                if tmp_net[i, j] < strength_thr:
                    continue
                circos.link(param1, param2, color=sector_colors[sector.name], 
                            direction=1, alpha=tmp_net[i, j]*alpha_scale, r1=65, r2=65)

    text_common_kws = dict(ha="left", va="center", size=12)
    circos.text("Ligand expression", r=94, color="black", **text_common_kws)
    circos.text("Receptor expression", r=80, color="black", **text_common_kws)
    circos.text("Lobe", r=68.5, color="black", **text_common_kws)
    circos.text(name, size=15, r=105)
    if is_show:
        fig = circos.plotfig(figsize=(8, 8))
    if save_name is not None:
        circos.savefig(save_name)
    pass

