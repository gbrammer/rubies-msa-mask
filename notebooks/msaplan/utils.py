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

            if (k == 0) & (add_labels):
                ax.text(
                    *sr.centroid[0],
                    f"Q{q}",
                    ha="center",
                    va="center",
                    **label_kwargs,
                )

            ax.plot(*sr.xy[0].T, alpha=0)

            if 1:
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

class MSATA(object):

    MSATA_MAG_LIMITS_MD = """
**Table 1.** Brightness ranges for NIRSpec MSATA filter and readout pattern options

|  Readout |  F110W  |  F140X  | CLEAR |  F110W_ | F140X_ | CLEAR_|
|:---------|:-------:|:-------:|:-----:|:-------:|:------:|:-----:|
|           |        |*S/N=20* |       |         | *Sat.* |       |
| NRSRAPID  | 22.0   | 23.0    | 23.8  | 19.5    | 20.6   | 21.3  |
| NRSRAPID1 |        |         | 24.5  |         |        | 21.9  |
| NRSRAPID2 |        |         | 24.9  |         |        | 22.9  |
| NRSRAPID6 | 24.0   | 25.0    | 25.7  | 21.3    | 22.3   | 23.1  |
    """

    THUMB_TEMPLATE = (
        "https://grizli-cutout.herokuapp.com/thumb?size=2.54&scl={scl:.1f}"
        "&default_filters=jwst&coords={ra:.6f},{dec:.6f}"
        "&asinh={asinh}&slit={ra:.6f},{dec:.6f},0.0,3.2,3.2"
    )

    def __init__(self, nrs=None, columns={}, **kwargs):
        """
        https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-msata-reference-star-selection-recommended-strategies#NIRSpecMSATAReferenceStarSelectionRecommendedStrategies-NIRSpecmagnitudes&gsc.tab=0
        """
        self.read_table_()
        
        self.nrs = nrs

        self.columns = columns
        self.parse_column_names()

    def parse_column_names(self):
        """
        """
        if not hasattr(self.nrs, "colnames"):
            return False

        cnames = {
            "id": ["ID"],
            "ra": ["RA"],
            "dec": ["Dec"],
            "mag": [
                "NRS_F110W", "NRS_F140X", "NRS_CLEAR", "Magnitude",
                "f115w_tot_1",
            ],
            "size": ["R50", "flux_radius", "size"],
            "ref": ["Reference", "is_star"],
        }
        
        for k in cnames:
            if k in self.columns:
                if self.columns[k] is not None:
                    continue

            self.columns[k] = None

            for c in cnames[k]:
                for func in [str.title, str.upper, str.lower]:
                    if func(c) in self.nrs.colnames:
                        self.columns[k] = func(c)
                        break
                
                if self.columns[k] is not None:
                    break

            if self.columns[k] is None:
                raise ValueError(f"No {k} in {cnames[k]} column found")

            if (k == "mag") & (self.columns[k] == "f115w_tot_1"):
                self.nrs["NRS_F110W"] = (
                    23.9 - 2.5 * np.log10(self.nrs["f115w_tot_1"])
                )
                self.columns["mag"] = "NRS_F110W"

    def jdocs_ranges(self):
        """
        Render markdown in Jupyter
        """
        from IPython.display import Markdown
        return Markdown(self.MSATA_MAG_LIMITS_MD)
        
    def read_table_(self):
        """
        Read markdown into a table
        """

        self.table = grizli.utils.GTable.read(
            self.MSATA_MAG_LIMITS_MD.strip(),
            format="ascii",
            delimiter="|",
            data_start=4,
            header_start=1
        )

        self.table = self.table[self.table.colnames[1:-1]]

        # Fill values in missing Readout / filter
        for c in ["F110W", "F140X"]:
            fill = self.table[c].mask & True
            dmag = np.mean((self.table[c] - self.table["CLEAR"])[~fill])
            self.table[c][fill] = (self.table["CLEAR"] + dmag)[fill]
            self.table[c + "_"][fill] = (self.table["CLEAR_"] + dmag)[fill]

    def select_mag_ranges(self, verbose=False):
        """
        """
        is_ref = self.nrs[self.columns["ref"]] > 0

        nrs_f110w = self.nrs[self.columns["mag"]]

        selection = {}

        if verbose:
            print(f"{'mode':^15} {'full':>7} {'& ref':>7}")
            print(f"{'-'*15:^15} {'-'*5:>7} {'-'*5:>7}")

        for i, row in enumerate(self.table):
            for j, c in enumerate("F110W F140X CLEAR".split()):
                in_range = (nrs_f110w < row[c]) & (nrs_f110w > row[c + "_"])
                key = f"{row['Readout']}_{c}".lower()
                
                if verbose:
                    print(
                        f"{key:>15} "
                        f"{in_range.sum():>7} {(in_range & is_ref).sum():>7}"
                    )
                
                selection[key] = in_range & is_ref
        
        return selection

    def select_sources(self, mode="nrsrapid_f110w", scl=2.0, asinh=False, nper=10, page=0, output="display", verbose=True, **kwargs):
        """
        """
        from IPython.display import display, Markdown

        cols = [self.columns[k] for k in ["id", "ra", "dec", "mag", "size"]]
        sub = self.nrs[cols][self.select_mag_ranges(verbose=False)[mode]]

        sl = slice(page * nper, (page + 1) * nper)
        if verbose:
            print(
                f"{mode}  [{sl.start}:{sl.stop}] {len(sub)} "
                + f"(page {page}/{len(sub)//nper})"
            )

        so = np.argsort(sub[self.columns["mag"]])[sl]
        sub = sub[so]

        sub[self.columns["mag"]].format = ".2f"
        sub[self.columns["size"]].format = ".1f"
        sub[self.columns["ra"]].format = ".6f"
        sub[self.columns["dec"]].format = ".6f"

        sub["Thumb_3.2"] = [
            "<img src=\""
            + self.THUMB_TEMPLATE.format(
                ra=row[self.columns["ra"]], dec=row[self.columns["dec"]],
                scl=scl, asinh=asinh,
            )
            + "\" width=200>"
            for row in sub
        ]

        if output == "display":
            df = sub.to_pandas()
            return display(Markdown(df.to_markdown()))
        else:
            return sub

    def plot_histogram(self, **kwargs):
        """
        Make a figure
        """
        nrs = self.nrs

        is_ref = nrs[self.columns["ref"]] > 0

        nrs_f110w = nrs[self.columns["mag"]][is_ref]

        fig, ax = plt.subplots(1, 1, figsize=(9,6))

        # R50 / flux_radius
        ax2 = ax.twinx()

        ax2.scatter(
            nrs_f110w,
            nrs[self.columns["size"]][is_ref],
            alpha=0.2,
            color="0.3",
        )
        ax2.set_ylim(0, 9.9)

        size_label = self.columns["size"] + ""
        if size_label == "R50":
            size_label = r"$R_{50}$ [pix]"
        elif size_label == "flux_radius":
            size_label = "flux_radius [pix]"

        ax2.set_ylabel(size_label)

        # Histogram
        _ = ax.hist(
            nrs_f110w,
            bins=np.arange(19.25, 26.01, 0.25),
            color="0.5", alpha=0.5
        )

        ymax = ax.get_ylim()[1] * 1.7
        ax.set_ylim(0, ymax)

        dy = 0.1
        y0 = 1.0 - dy * 4.5
        fs = 7
        yw = 1.0 / 4

        tkwargs = {
            "fontsize": 7,
            "zorder": 10,
            "color": "w",
            "weight": "bold",
        }

        for i, row in enumerate(self.table):
            c_i = plt.cm.Spectral(i / 3.)
            yj = y0 + i * dy
            for j, c in enumerate("F110W F140X CLEAR".split()):
                yj += dy / 3.

                in_range = (nrs_f110w < row[c]) & (nrs_f110w > row[c + "_"])
                n_j = in_range.sum()

                ty = (yj + (0.5 * yw - 0.015) * dy) * ymax

                # Count
                ax.text(
                    row[c + "_"] + 0.05,
                    ty,
                    f"{n_j}",
                    ha="left", va="center", **tkwargs
                )
                
                # Filter
                ax.text(
                    row[c] - 0.05,
                    ty, c, ha="right", va="center", **tkwargs
                )

                # Readout
                if j == 1:
                    ax.text(
                        np.mean([row[c], row[c + "_"]]),
                        ty,
                        row["Readout"],
                        ha="center",
                        va="center",
                        **tkwargs
                    )

                ax.fill_between(
                    [row[c], row[c + "_"]],
                    np.ones(2) * yj * ymax,
                    np.ones(2) * (yj + 1. / 4 * dy) * ymax,
                    alpha=0.95,
                    color=c_i,
                    zorder=tkwargs["zorder"] - 1,
                )

        # Total
        ax.text(
            0.5, 0.05,
            "".join([
                f"'{self.columns['ref']}': ",
                r"$N_\mathrm{tot}$ = ",
                f"{is_ref.sum()}"
            ]),
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=tkwargs["fontsize"] * 1.3,
            weight=tkwargs["weight"],
            color="k",
        )

        ax.grid(zorder=tkwargs["zorder"] - 5)

        _ = ax.set_xlabel(f"{self.columns['mag']} [mag]")
        _ = ax.set_ylabel(r"$N$ / 0.25 mag")

        fig.tight_layout(pad=1)

        return fig
    