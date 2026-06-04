import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import grizli.utils

PRIORITY_COLOR = {
    0: 'magenta',
    1: 'tomato',
    2: 'orange',
    3: 'steelblue',
    # 4:'purple',
   11: 'olive',
   'other': '0.7'
}
                  
def data_path(file=None, raise_exception=True):

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if file is None:
        return data_dir

    file_path = None
    for path in ["", data_dir]:
        file_path_ = os.path.join(path, file)
        if os.path.exists(file_path_):
            file_path = file_path_

    if file_path is None:
        if raise_exception:
            msg = f"No such file or directory: '{file_path_}'"
            raise FileNotFoundError(msg)
        else:
            file_path = file_path_

    return file_path


def shutter_mask_footprint():
    """
    Get shutter mask in V2,V3 coordinates
    """
    mask_file = data_path(file="shutter_mask.yml")

    with open(mask_file) as fp:
        mask_data = yaml.load(fp, Loader=yaml.Loader)

    mask = {}
    for label in mask_data:
        mask[label] = {}
        for q in mask_data[label]:
            mask[label][q] = grizli.utils.SRegion(mask_data[label][q], wrap=False)

    return mask


def show_msa_layout(
    figsize=(6, 6),
    ax=None,
    siaf_aper=None,
    legend=True,
    ascale=1.0,
    add_labels=True,
    show_footprints=["all", "g395m", "prism"],
    label_kwargs={"alpha": 1.0, "color": "0.5", "zorder": 101, "fontsize": 7},
    **kwargs,
):
    """
    Make a plot showing the MSA footprint
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    if siaf_aper is None:
        ax.set_xlim(216, 538)
        ax.set_ylim(-591, -267)
        ax.set_aspect(1)

    fc = ["None", "tomato", "None"]
    ec = ["goldenrod", "tomato", "darkred"]
    # labels = ['All', 'G395M', 'PRISM']
    alphas = [0.8, 0.5, 0.3]
    hatch = [None, None, "|||||"]

    sfoot = shutter_mask_footprint()

    for k, label in enumerate(sfoot):
        for q in sfoot[label]:
            sr = sfoot[label][q]
            if siaf_aper is not None:
                rd = siaf_aper.tel_to_sky(*sr.xy[0].T)
                sr = grizli.utils.SRegion(np.array(rd))

            if (k == 0) & (add_labels) & (label in show_footprints):
                ax.text(
                    *sr.centroid[0],
                    f"Q{q}",
                    ha="center",
                    va="center",
                    **label_kwargs,
                )

            ax.plot(*sr.xy[0].T, alpha=0)

            if (label in show_footprints):
                sr.add_patch_to_axis(
                    ax,
                    ec=ec[k],
                    fc=fc[k],
                    alpha=alphas[k] * ascale,
                    hatch=hatch[k],
                    label=(label if q == 1 else None),
                )

    # show dispersion direction
    disp_vector = np.array([[378.56, 390.11], [-428.4, -418.3]])

    if siaf_aper is not None:
        disp_vector = np.array(siaf_aper.tel_to_sky(*disp_vector)).T
    else:
        disp_vector = disp_vector.T

    if fig is not None:
        if add_labels:

            qkwargs = {"alpha": 1.0, "color": "0.5", "zorder": 101}
            for k in label_kwargs:
                if k in qkwargs:
                    qkwargs[k] = label_kwargs[k]

            ax.quiver(
                *disp_vector[0],
                *(disp_vector - disp_vector[0])[1],
                angles="xy",
                units="xy",
                width=(1.09 if siaf_aper is None else 1.0 / 3600),
                headwidth=5,
                headlength=7,
                **qkwargs,
            )

            ax.text(
                *(disp_vector[0] - 0.4 * np.diff(disp_vector, axis=0))[0],
                r"$\lambda_+$",
                ha="center",
                va="center",
                **label_kwargs,
            )

        if legend:
            ax.legend(loc="lower right")

        ax.grid()

        if siaf_aper is None:
            ax.set_xlabel("V2")
            ax.set_ylabel("V3")
        else:
            ax.set_xlabel("R.A.")
            ax.set_ylabel("Dec.")
            ax.set_aspect(1.0 / np.cos(sr.centroid[0][1] / 180 * np.pi))
            ax.set_xlim(*ax.get_xlim()[::-1])

        fig.tight_layout()

    return fig


def load_slit_transformation_coeffs():
    """
    Load the transformation polynomials:

    >>> v2 = coeffs[q][0](row, col)
    >>> v3 = coeffs[q][1](row, col)

    """
    from astropy.modeling.polynomial import Polynomial2D

    tab = grizli.utils.read_catalog(data_path(file="all_grating_coeffs.2026.3.1.fits"))

    all_coeffs = {}
    all_inv_coeffs = {}

    array_data = []
    for c in tab.colnames:
        if c.startswith("c"):
            array_data.append(tab[c])

    array_data = np.array(array_data)

    for i, row in enumerate(tab):
        g = row["grating"]
        if g not in all_coeffs:

            all_coeffs[g] = {}
            all_inv_coeffs[g] = {}

            for q in [1, 2, 3, 4]:
                all_coeffs[g][q] = []
                all_inv_coeffs[g][q] = []

        p2d = Polynomial2D(degree=3)
        p2d.parameters += array_data[:, i]

        q = row["quadrant"]
        if row["inv"]:
            all_inv_coeffs[g][q].append(p2d)
        else:
            all_coeffs[g][q].append(p2d)

    return all_coeffs, all_inv_coeffs


def replace_strings_in_file(file, repl, overwrite=True):
    """
    Replace all strings in a file
    """
    
    with open(file, "r") as fp:
        data = fp.read()

    for s1, s2 in np.atleast_2d(repl):
        data = data.replace(s1, s2)

    if overwrite:
        with open(file, "w") as fp:
            fp.write(data)

    return data


def set_axis_ticks(ax=None, which='xy'):
    if ax is None:
        ax = plt.gca()

    dxcen = 450
    dycen = 231.7

    xlim = (377 + dxcen + 20, -20)
    ylim = (171 + dycen + 10, -10)

    if 'x' in which:
        xt = [1, 90, 180, 270, 377]
        xtv = xt + [xi + dxcen for xi in xt]
        ax.set_xticks(xtv)
        ax.set_xticklabels(xt*2)
        ax.set_xlabel('xcen')
        ax.set_xlim(*xlim)

    if 'y' in which:
        xt = [1, 60, 120, 171]
        xtv = xt + [xi + dycen for xi in xt]
        ax.set_yticks(xtv)
        ax.set_yticklabels(xt*2)
        ax.set_ylabel('ycen')
        ax.set_ylim(*ylim)

    return ax
    

######
def msa_quadrant_footprint(plan_file='uds_obs4a_prism.json', ax=None, patch_kwargs={'fc':'magenta', 'alpha': 0.5}, ec=["None","None","magenta","None"]):
    import pysiaf
    from pysiaf.utils import rotations
    from grizli import utils
    import json

    print(plan_file)
    
    with open(plan_file) as fp:
        plan = json.load(fp)

    siaf_nrs = pysiaf.Siaf('NIRSpec')
    ap0 = siaf_nrs['NRS_FULL_MSA']

    exp = plan['configs'][0]['exposures'][0]
    
    att = rotations.attitude(
        ap0.V2Ref, ap0.V3Ref,
        exp['ra'],
        exp['dec'],
        plan['aperturePA'] - ap0.V3IdlYAngle
    )

    corners = []
    
    for q in [1,2,3,4]:
        ap = siaf_nrs[f'NRS_VIGNETTED_MSA{q}']
        ap.set_attitude_matrix(att)
        cx, cy = ap.corners('sky')
        corners.append((np.roll(cx, -1), np.roll(cy, -1)))

    if ax is not None:
        for i, c in enumerate(corners):
            sr = utils.SRegion(np.array(c))
            patch_kwargs['ec'] = ec[i]
            sr.add_patch_to_axis(ax, **patch_kwargs)
            
    return plan, corners

