import os
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from grizli import utils

from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator

import astropy.units as u
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.coordinates import SkyCoord, Angle

from . import utils as plan_utils
from .utils import PRIORITY_COLOR

WAVES = np.linspace(0.55, 5.5, 512)

VERBOSITY = True


def wscale_(waves, w0, wlim):
    """
    Renormalize wavelength for polynomials
    """
    ww = np.log(waves / w0)
    ww = (ww - wlim[0]) / (wlim[1] - wlim[0]) * 2 - 1
    return ww


class PrismTrace:
    """
    Helper function for getting the *rough* trace as a function of wavelength for
    an arbitrary slitlet by interpolating the trace from nearby measured slitlets
    """

    def __init__(self):

        self.tab = utils.read_catalog(
            plan_utils.data_path(file="slit_parameters_prism.fits")
        )

        self.setup_coeffs()

    def setup_coeffs(self):
        """
        Trace is x = np.polyval(cx, wcsale(wave, w0, wlim))
        """
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LinearLSQFitter

        tab = self.tab

        # Polynomial coefficients
        self.cx = np.array([tab[f"cx{i}"] for i in range(tab.meta["NX"] + 1)])
        self.cy = np.array([tab[f"cy{i}"] for i in range(tab.meta["NY"] + 1)])

        self.wxlim = [tab.meta["WXLIM0"], tab.meta["WXLIM1"]]
        self.wylim = [tab.meta["WYLIM0"], tab.meta["WYLIM1"]]
        self.wx = tab.meta["WX"]
        self.wy = tab.meta["WY"]

        # Fit 2D polynomials
        fit = LinearLSQFitter()
        poly = Polynomial2D(degree=3)

        xpoly = {}
        ypoly = {}

        for det in ["NRS1", "NRS2"]:
            in_det = self.tab["detector"] == det
            quads = np.unique(self.tab["shutter_quadrant"][in_det])
            for q in quads:
                iq = in_det & (self.tab["shutter_quadrant"] == q)
                key = f"{det}-{q}"

                cx = [
                    fit(
                        poly,
                        self.tab["shutter_row"][iq],
                        self.tab["shutter_column"][iq],
                        self.cx[i, iq],
                    )
                    for i in range(self.cx.shape[0])
                ]
                cy = [
                    fit(
                        poly,
                        self.tab["shutter_row"][iq],
                        self.tab["shutter_column"][iq],
                        self.cy[i, iq],
                    )
                    for i in range(self.cy.shape[0])
                ]

                xpoly[key] = cx
                ypoly[key] = cy

        self.xpoly = xpoly
        self.ypoly = ypoly

    def get_trace(
        self,
        waves=np.linspace(0.6, 5.3, 1024),
        shutter_quadrant=1,
        shutter_row=333,
        shutter_column=112,
        **kwargs,
    ):
        """
        Get the trace of a given shutter

        Returns
        -------
        drowm, dcolm : int, int
            Offset to nearest row, column in the table

        rowi, coli : int, int
            Row, column of nearest row in the table

        ptraces : (xp, yp)
        """
        drow = self.tab["shutter_row"] - shutter_row
        dcol = self.tab["shutter_column"] - shutter_column

        dr = np.sqrt((drow * 1.0) ** 2 + (dcol * 2.5) ** 2)

        in_quad = self.tab["shutter_quadrant"] == shutter_quadrant

        ix = np.argmin(dr[in_quad])
        drmin = dr[in_quad][ix]
        drowm = drow[in_quad][ix]
        dcolm = dcol[in_quad][ix]

        rowi = self.tab["shutter_row"][in_quad][ix]
        coli = self.tab["shutter_column"][in_quad][ix]

        traces = []
        ptraces = []

        for det in ["NRS1", "NRS2"]:
            mat = in_quad
            mat &= (self.tab["shutter_row"] == rowi) & (
                self.tab["shutter_column"] == coli
            )
            mat &= self.tab["detector"] == det

            #             if mat.sum() == 0:
            #                 traces.append(None)
            #                 ptraces.append(None)

            #                 continue

            #             ix = np.where(mat)[0][0]

            key = f"{det}-{shutter_quadrant}"
            if key not in self.xpoly:
                traces.append(None)
                ptraces.append(None)
                continue

            cx = [p(shutter_row, shutter_column) for p in self.xpoly[key]]
            cy = [p(shutter_row, shutter_column) for p in self.ypoly[key]]
            xp = np.polyval(cx, wscale_(waves, self.wx, self.wxlim)) - 1
            yp = np.polyval(cy, wscale_(waves, self.wy, self.wylim)) - 1
            ptraces.append((xp, yp))

            traces.append((xp, yp))

        return drowm, dcolm, rowi, coli, ptraces, traces

    def slit_exposure_map(
        self,
        waves=np.linspace(0.6, 5.3, 1024),
        shutter_quadrant=1,
        shutter_row=333,
        shutter_column=112,
        shutters=[-1, 0, 1],
        **kwargs,
    ):
        """
        Make an exposure map assuming that each 2D spectrum subtends 5 pixels per shutter
        """

        _, _, _, _, ptraces, traces = self.get_trace(
            waves=waves,
            shutter_quadrant=shutter_quadrant,
            shutter_row=shutter_row,
            shutter_column=shutter_column,
        )

        arrs = [np.zeros((2048, 2048)) for i in range(2)]
        for i, tr in enumerate(ptraces):
            if tr is None:
                continue

            xp, yp = tr
            xpi = np.cast[int](np.round(xp))
            ypi = np.cast[int](np.round(yp))

            ok = (xpi >= 0) & (xpi < 2048) & (ypi >= 0) & (ypi < 2048)
            if ok.sum() > 0:
                for xi, yi in zip(xpi[ok], ypi[ok]):
                    for s in shutters:
                        for dy in range(-2, 3):
                            arrs[i][yi + dy + s * 5, xi] = 1

        return arrs

    def plot_shutter_table(
        self,
        waves,
        shutter_table,
        figsize=(8, 10),
        plot_kws=dict(alpha=0.3, linewidth=2),
    ):
        """ """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

        shutter_table["wmin"] = -1.0
        shutter_table["wmax"] = -1.0
        shutter_table["detector"] = 0

        for j, shutter_def in enumerate(shutter_table):
            drowm, dcolm, rowi, coli, ptraces, traces = pr.get_trace(
                waves=waves, **shutter_def
            )

            for i, tr in enumerate(ptraces):
                if tr is not None:
                    # label = 'Q={shutter_quadrant} R={shutter_row} C={shutter_column}'.format(**args)
                    axes[i].plot(
                        *tr,
                        color=plt.cm.jet(shutter_def["shutter_quadrant"] / 4.0),
                        **plot_kws,
                    )
                    trx = tr[0]
                    ok = (trx > 0) & (trx < 2047)
                    if ok.sum() > 2:
                        shutter_table["detector"][j] += 2**i
                        shutter_table["wmin"][j] = waves[ok].min()
                        shutter_table["wmax"][j] = waves[ok].max()

        for ax in axes:
            ax.grid()

        axes[0].set_title("NRS1")
        axes[1].set_title("NRS2")

        axes[0].set_xlim(-5, 2048 + 5)
        axes[0].set_ylim(-5, 2048 + 5)

        axes[0].set_xlabel("x pixel")
        axes[0].set_ylabel("y pixel")

        fig.tight_layout(pad=1)

        return fig

    def slitlet_path(
        self,
        waves=None,
        shutter_quadrant=3,
        shutter_row=3,
        shutter_column=80,
        pad=1,
        **kwargs,
    ):
        """
        Get a path object
        """
        from matplotlib.path import Path

        args = dict(
            shutter_quadrant=shutter_quadrant,
            shutter_row=shutter_row,
            shutter_column=shutter_column + 1,
        )
        drowm, dcolm, rowi, coli, upper, traces = self.get_trace(waves=waves, **args)

        args = dict(
            shutter_quadrant=shutter_quadrant,
            shutter_row=shutter_row,
            shutter_column=shutter_column - 1,
        )
        drowm, dcolm, rowi, coli, lower, traces = self.get_trace(waves=waves, **args)

        pa = []
        for i, tr in enumerate(upper):
            if tr is not None:
                dy = (upper[i][1] - lower[i][1]) / 4
                y0 = (upper[i][1] + lower[i][1]) / 2.0
                px = upper[i][0]
                if (px.min() > 2048) | (px.max() < 0):
                    pai = None
                else:
                    pai = Path(
                        np.array(
                            [
                                np.append(px, px[::-1]),
                                np.append(
                                    y0 + (1 + 2 * pad) * dy - 0.5,
                                    (y0 - (1 + 2 * pad) * dy + 0.5)[::-1],
                                ),
                            ]
                        ).T
                    )
                    test = (px > 0) & (px < 2048)
                    pai.frac = test.sum() / len(test)
            else:
                pai = None

            pa.append(pai)

        return pa

    def plot_single_trace(self, q=3, row=259, col=40, detector="NRS1", waves=WAVES):

        from matplotlib.ticker import MultipleLocator, AutoMinorLocator

        args = dict(shutter_quadrant=q, shutter_row=row, shutter_column=col)

        _ = self.get_trace(waves=waves, **args)
        drowm, dcolm, rowi, coli, ptraces, traces = _

        det_i = ["NRS1", "NRS2"].index(detector.upper())

        xp, yp = ptraces[det_i]
        xpi = np.arange(xp.min(), xp.max(), 0.5)

        waves = np.interp(xpi, xp, waves)

        TRACE_WAVES = waves * 1

        # Plot a shutter
        expmap = self.slit_exposure_map(waves=TRACE_WAVES, shutters=[0], **args)

        # Three shutters
        exp3 = self.slit_exposure_map(waves=TRACE_WAVES, shutters=[-1, 0, 1], **args)

        pa = self.slitlet_path(waves=TRACE_WAVES, pad=1, **args)
        pa0 = self.slitlet_path(waves=TRACE_WAVES, pad=0, **args)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.imshow(
            expmap[det_i] + exp3[det_i],
            aspect="auto",
            cmap="gray_r",
            extent=(0, 2047, 0, 2047),
        )

        ax.plot(xp, yp)
        ax.fill_between(
            [-400, -1],
            np.ones(2) - 1000,
            np.ones(2) + 3000,
            color="0.8",
            hatch="//////",
            alpha=0.1,
        )
        ax.fill_between(
            [2048, 2500],
            np.ones(2) - 1000,
            np.ones(2) + 3000,
            color="0.8",
            hatch="//////",
            alpha=0.1,
        )

        ax.set_xlim(xp.min() - 50, xp.max() + 50)
        ax.set_ylim(yp.min() - 10, yp.max() + 10)
        # ax.set_xlim(750,1400)
        # ax.set_ylim(1834, 1856)

        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.grid()

        ax.add_patch(PathPatch(pa[det_i], fc="None", ec="r"))
        ax.add_patch(PathPatch(pa0[det_i], fc="None", ec="magenta"))

        ax.set_xlabel(f"x pixel ({detector})")
        ax.set_ylabel(f"y pixel")

        ax.set_title(f"Q={q} row={row} col={col}")

        fig.tight_layout()
        return fig


class EmpiricalTrace:

    def __init__(
        self, stab=None, grating="G395M", filter="F290LP", skip_ids=[], **kwargs
    ):
        """
        Refined grating traces for a calculated shutter table with traces based
        on extracted spectra

        stab : msaplan.apt.ShutterTable
        """
        self.stab = stab
        self.grating = grating
        self.filter = filter
        self.skip_ids = skip_ids

        # Single MSA coordinate frame
        self.dxcen = 450
        self.dycen = 230

        self.xquad = (
            self.stab.shutter_table["shutter_row"]
            + np.isin(self.stab.shutter_table["shutter_quadrant"], [3, 4]) * self.dxcen
        )

        self.yquad = (
            self.stab.shutter_table["shutter_column"]
            + np.isin(self.stab.shutter_table["shutter_quadrant"], [2, 4]) * self.dycen
        )

        self.build_traces(**kwargs)

    def build_traces(
        self,
        pad_size={0: 12.5, 1: 7.5},
        default_pad=7.5,
        stuck_open_pad=2.5,
        step=16,
        cstep=128,
        ncoarse=32,
        **kwargs,
    ):
        """
        Build grating trace paths
        """
        self.slit_meta = utils.read_catalog(
            plan_utils.data_path("slit_cutout_metadata_202504.fits")
        )

        test = np.isin(
            self.slit_meta["grating"].data.astype(str),
            ["PRISM", "G395M", "G235M"]
        )
        test &= self.slit_meta["detector"] == "NRS1"
        test &= np.isin(self.slit_meta["quadrant"], [1,2])
        test &= self.slit_meta["xcen"] < 60
        self.slit_meta = self.slit_meta[~test]

        self.pad_size = pad_size
        self.default_pad = default_pad
        self.stuck_open_pad = stuck_open_pad

        # Parameters to interpolate
        params = [
            "sltstrt1",
            "sltsize1",
            "sltstrt2",
            "sltsize2",
            "trace_c0",
            "trace_c1",
            "trace_c2",
            "x_min",
            "x_max",
            "wave_min",
            "wave_max",
        ]

        msg = f"Optimize grating traces for for {self.grating}-{self.filter}"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        N = len(self.stab.shutter_table)
        No = len(self.stab.smask.stuck_open_table)

        grating_paths = [[None, None] for i in range(N)]
        coarse_grating_paths = [[None, None] for i in range(N)]
        so_grating_paths = [[None, None] for i in range(No)]

        # step = 16
        # cstep = 128

        for i, det in enumerate(["NRS1", "NRS2"]):

            test = self.slit_meta["grating"] == self.grating
            test &= self.slit_meta["filter"] == self.filter
            test &= self.slit_meta["detector"] == det
            test &= (self.slit_meta["x_max"] - self.slit_meta["x_min"]) > 4

            sub_meta = self.slit_meta[test]
            # print("xxx", det, test.sum())

            qs = utils.Unique(sub_meta["quadrant"], verbose=False)
            qx = utils.Unique(
                self.stab.shutter_table["shutter_quadrant"], verbose=False
            )
            qo = utils.Unique(
                self.stab.smask.stuck_open_table["Q"], verbose=False
            )

            suffix = f'_{det}_{self.grating}'.lower()
            for p in params:
                self.stab.shutter_table[p] = np.nan
                self.stab.shutter_table[p + suffix] = np.nan
                self.stab.smask.stuck_open_table[p] = np.nan

            for q in qx.values:
                qsi = qs[q]
                qxi = np.where(qx[q])[0]
                if qsi.sum() < 3:
                    continue

                qoi = qo[q]

                coo = np.array([sub_meta[qsi][c] for c in ["xcen", "ycen"]]).T

                cx = np.array([
                    self.stab.shutter_table[c][qxi]
                    for c in ["shutter_row", "shutter_column"]
                ])

                co = np.array([
                    self.stab.smask.stuck_open_table[c][qoi]
                    for c in ["x", "y"]
                ])

                for p in params:
                    interp = LinearNDInterpolator(
                        coo, sub_meta[p][qsi], fill_value=np.nan
                    )
                    value_ = interp(*cx)
                    valid_ = np.isfinite(value_) # | True
                    self.stab.shutter_table[p][qxi[valid_]] = value_[valid_]
                    self.stab.shutter_table[p + suffix][qxi[valid_]] = value_[valid_]
                    self.stab.smask.stuck_open_table[p][qoi] = interp(*co)
                    
                    if p in ["wave_min", "wave_max", "x_min", "x_max"]:
                        self.stab.shutter_table[p].format = ".2f"
                        self.stab.shutter_table[p + suffix].format = ".2f"

            miss = ~np.isfinite(self.stab.shutter_table["x_min"])
            miss |= (
                self.stab.shutter_table["x_max"] - self.stab.shutter_table["x_min"]
            ) < 4

            for p in ["wave_min", "wave_max", "x_min", "x_max"]:
                self.stab.shutter_table[p][miss] = np.nan
                self.stab.shutter_table[p + suffix][miss] = np.nan

            for j in np.where(~miss)[0]:
                row = self.stab.shutter_table[j]
                # xp = np.arange(int(row['x_min']), int(row['x_max']), step)
                xwidth = row["x_max"] - row["x_min"]
                xp = np.linspace(row["x_min"], row["x_max"], int(xwidth / step))

                dx = xp - (row["sltsize1"] / 2)
                dy = (
                    dx**0 * row["trace_c2"]
                    + dx**1 * row["trace_c1"]
                    + dx**2 * row["trace_c0"]
                )

                xdet = row["sltstrt1"] - 1 + xp
                ydet = row["sltstrt2"] - 1 + dy

                p_i = int(np.floor(row["priority"]))
                if p_i in pad_size:
                    pad_i = pad_size[p_i]
                else:
                    pad_i = default_pad

                grating_paths[j][i] = Path(
                    np.array(
                        [
                            np.append(xdet, xdet[::-1]),
                            np.append(ydet - pad_i, ydet[::-1] + pad_i),
                        ]
                    ).T
                )

                # xp = np.arange(int(row['x_min']), int(row['x_max']), cstep)
                xp = np.linspace(row["x_min"], row["x_max"], ncoarse)

                dx = xp - (row["sltsize1"] / 2)
                dy = (
                    dx**0 * row["trace_c2"]
                    + dx**1 * row["trace_c1"]
                    + dx**2 * row["trace_c0"]
                )

                xdet = row["sltstrt1"] - 1 + xp
                ydet = row["sltstrt2"] - 1 + dy

                coarse_grating_paths[j][i] = Path(
                    np.array(
                        [
                            np.append(xdet, xdet[::-1]),
                            np.append(ydet - pad_i, ydet[::-1] + pad_i),
                        ]
                    ).T
                )

            for j in range(No):
                row = self.stab.smask.stuck_open_table[j]
                try:
                    xp = np.arange(int(row["x_min"]), int(row["x_max"]), step)
                except:
                    continue

                dx = xp - (row["sltsize1"] / 2)
                dy = (
                    dx**0 * row["trace_c2"]
                    + dx**1 * row["trace_c1"]
                    + dx**2 * row["trace_c0"]
                )

                xdet = row["sltstrt1"] - 1 + xp
                ydet = row["sltstrt2"] - 1 + dy

                so_grating_paths[j][i] = Path(
                    np.array(
                        [
                            np.append(xdet, xdet[::-1]),
                            np.append(
                                ydet - stuck_open_pad, ydet[::-1] + stuck_open_pad
                            ),
                        ]
                    ).T
                )

        # overlaps
        grating_olap = np.zeros((self.stab.N, self.stab.N), dtype=bool)

        if self.stab.N > 400:
            _iter = tqdm(range(self.stab.N))
        else:
            _iter = range(self.stab.N)

        for i in _iter:
            pi = coarse_grating_paths[i]
            for j in range(i + 1, self.stab.N):
                pj = coarse_grating_paths[j]
                for k in [0, 1]:
                    if (pi[k] is not None) & (pj[k] is not None):
                        grating_olap[i, j] |= pi[k].intersects_path(pj[k])
                        grating_olap[j, i] |= grating_olap[i, j]

        # Stuck open shutters
        stuck_olap = np.zeros(self.stab.N, dtype=bool)

        if self.stab.N > 400:
            _iter = tqdm(range(self.stab.N))
        else:
            _iter = range(self.stab.N)

        for i in _iter:
            pi = grating_paths[i]
            for pj in so_grating_paths:
                for k in [0, 1]:
                    if (pi[k] is not None) & (pj[k] is not None):
                        stuck_olap[i] |= pi[k].intersects_path(pj[k])

        self.grating_paths = grating_paths
        self.coarse_grating_paths = coarse_grating_paths
        self.so_grating_paths = so_grating_paths

        self.grating_olap = grating_olap
        self.stuck_olap = stuck_olap

    def add_with_shutter_padding(
        self,
        selection,
        allocated=None,
        xpad=7,
        ypad=2,
        ypad_allocated=2,
        allow_overlap=True,
        randomize=False,
        **kwargs,
    ):
        """
        Add objects to the list if shutters offset by a minimum distance
        from already-allocated shutters
        """
        if allocated is None:
            allocated = np.zeros(self.stab.N, dtype=bool)

        ix = np.where(selection)[0]
        weight = self.stab.cat[self.stab.weight_column][self.stab.ix]
        wi = weight[ix]

        if randomize:
            so = np.argsort(np.random.normal(size=len(wi)))
        else:
            so = np.argsort(wi)[::-1]

        added = allocated & False

        if not allow_overlap:
            olap = (self.grating_olap * (allocated | added)).sum(axis=1)

        for j in so:
            i = ix[j]
            if not allow_overlap:
                if olap[i] > 0:
                    continue

            touches = 0
            for sub, ypad_ in zip([allocated, added], [ypad_allocated, ypad]):
                if sub.sum() > 0:
                    dxq = self.xquad[i] - self.xquad[sub]
                    dyq = self.yquad[i] - self.yquad[sub]
                    touches += ((np.abs(dxq) < xpad) & (np.abs(dyq) < ypad_)).sum()

            if touches > 0:
                debug = (
                    f"j={j} wi[j]={wi[j]} i={i} "
                    f"p_i={self.stab.shutter_table['priority'][i]} "
                    f"xquad[i]={self.xquad[i]} yquad[i]={self.yquad[i]}"
                    f" min(dxq)={np.abs(dxq).min()}"
                    f" min(dyq)={np.abs(dyq).min()}"
                    f" n_touches={touches}"
                )
                utils.log_comment(utils.LOGFILE, debug, verbose=(VERBOSITY > 1))

            added[i] = touches == 0
            if added[i] & (not allow_overlap):
                # Update overlap list
                olap = (self.grating_olap * (allocated | added)).sum(axis=1)

        return added

    def allocate_shutters(
        self,
        xpad=7,
        ypad=2,
        ypad_allocated=2,
        any_p0=True,
        force_p0_ids=[],
        filler_priorities=[3, 4, 5, 6, 7, 8],
        filler_xpad=20,
        allow_primary_overlap=True,
        allow_filler_overlap=True,
        allow_primary_stuck_open=True,
        allow_filler_stuck_open=False,
        randomize_fillers=False,
        randomize_min_extra=1,
        initial_mask=None,
        **kwargs,
    ):
        """ """
        clean_mask = np.zeros(self.stab.N, dtype=bool)
        if initial_mask is not None:
            if isinstance(initial_mask, str):
                initial_mask = self.clean_prism_mask(
                    force_p0_ids=force_p0_ids,
                    allow_filler_stuck_open=allow_filler_stuck_open
                )
                
                msg = (
                    "allocate_shutters: initial mask from prism\n"
                    f"allocate_shutters: orig={self.stab.ok_prism.sum()}"
                    f" clean_primary={initial_mask.sum()}"
                )
                utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

            clean_mask |= initial_mask

        is_p0 = self.stab.shutter_table["priority"] == 0

        is_p0 |= np.isin(self.stab.shutter_table["id"], force_p0_ids)

        is_p1 = (~is_p0) & (self.stab.shutter_table["priority"] == 1)

        is_p12 = (~is_p0) & (self.stab.shutter_table["priority"] < 3)

        is_p3 = (~is_p0) & (self.stab.shutter_table["priority"] <= 3.99)

        is_filler = np.isin(self.stab.shutter_table["priority"], filler_priorities)

        # Force P0
        test = is_p0 & True
        test &= ~clean_mask

        if any_p0:
            add_p0 = test & True
        else:
            add_p0 = self.add_with_shutter_padding(
                selection=test,
                xpad=xpad,
                ypad=ypad,
                ypad_allocated=ypad_allocated,
                allocated=clean_mask,
                allow_overlap=True,
            )
        
        clean_mask |= add_p0
        
        msg = f"     P0: {add_p0.sum():>9} shutters (extra id={force_p0_ids})"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        skip = np.isin(self.stab.shutter_table["id"], self.skip_ids)

        # P1 that doesn't overlap with P0
        test = is_p12 & True
        test &= (self.grating_olap * is_p0).sum(axis=1) == 0
        test &= ~clean_mask
        test &= ~skip
        if not allow_primary_stuck_open:
            test &= ~self.stuck_olap

        add_p12 = self.add_with_shutter_padding(
            selection=test,
            xpad=xpad,
            ypad=ypad,
            ypad_allocated=ypad_allocated,
            allocated=clean_mask,
            allow_overlap=allow_primary_overlap,
        )

        msg = f"   P1,2: {add_p12.sum():>3} / {test.sum():>3} additional shutters"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        clean_mask |= add_p12

        # P3 that doesn't overlap with P01
        test = is_p3 & True
        test &= (self.grating_olap * (is_p0 | is_p1)).sum(axis=1) == 0
        test &= ~clean_mask
        test &= ~skip
        if not allow_primary_stuck_open:
            test &= ~self.stuck_olap

        add_p3 = self.add_with_shutter_padding(
            selection=test,
            xpad=xpad,
            ypad=ypad,
            ypad_allocated=ypad_allocated,
            allocated=clean_mask,
            allow_overlap=allow_primary_overlap,
        )

        # otest = add_to_sample(test=test, pad=pad)
        msg = f"     P3: {add_p3.sum():>3} / {test.sum():>3} additional shutters"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        clean_mask |= add_p3

        # Fillers that don't overlap with anything already allocated
        test = (self.grating_olap * (is_p0 | is_p3)).sum(axis=1) == 0
        test &= ~clean_mask
        test &= ~skip
        test &= is_filler
        if not allow_filler_stuck_open:
            test &= ~self.stuck_olap

        kwargs = dict(
            selection=test,
            xpad=filler_xpad,
            ypad=ypad,
            ypad_allocated=ypad_allocated,
            allocated=clean_mask,
            allow_overlap=allow_filler_overlap,
        )

        add_fillers = self.add_with_shutter_padding(**kwargs)

        if randomize_fillers:
            random_counts = np.zeros(randomize_fillers * 1, dtype=int)
            random_weights = np.zeros(randomize_fillers * 1, dtype=int)

            for i in tqdm(range(randomize_fillers * 1)):
                np.random.seed(i)
                fillers_ = self.add_with_shutter_padding(
                    randomize=True,
                    **kwargs
                )
                random_counts[i] = fillers_.sum()
                ix_ = self.stab.shutter_table["ix"][fillers_]
                random_weights[i] = self.stab.cat["Priority"][ix_].sum()
                
            msg = f"Fillers: {randomize_fillers * 1} random "
            msg += f" N={add_fillers.sum()} min={random_counts.min()} max={random_counts.max()}"
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

            random_imax = np.nanargmin(
                random_weights * np.nan**(random_counts < random_counts.max())
            )
            
            # Random got more, so use that
            if random_counts[random_imax] > (add_fillers.sum() + randomize_min_extra):
                np.random.seed(random_imax)

                add_fillers = self.add_with_shutter_padding(
                    randomize=True,
                    **kwargs
                )

        else:
            random_imax = None
            random_counts = random_weights = None

        msg = f"Fillers: {add_fillers.sum():>3} / {test.sum():>3} additional shutters"
        msg += f" priority={filler_priorities}"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        clean_mask |= add_fillers

        print("\nTotal shutters: ", clean_mask.sum())

        result = {
            "xpad": xpad,
            "ypad": ypad,
            "ypad_allocated": ypad_allocated,
            "filler_xpad": filler_xpad,
            "filler_priorities": filler_priorities,
            "force_p0_ids": force_p0_ids,
            "add_p0": add_p0,
            "add_p12": add_p12,
            "add_p3": add_p3,
            "add_fillers": add_fillers,
            "randomize_fillers": randomize_fillers,
            "randomize_min_extra": randomize_min_extra,
            "random_imax": random_imax,
            "random_counts": random_counts,
            "random_weights": random_weights,
            "allow_primary_stuck_open": allow_primary_stuck_open,
            "allow_filler_stuck_open": allow_filler_stuck_open,
            "clean_mask": clean_mask,
        }

        return result

    def clean_prism_mask(self, force_p0_ids=[], allow_filler_stuck_open=False, **kwargs):
        """
        Remove entries from the prism mask whose grating spectra overlap with
        primaries
        """
        is_p0 = self.stab.shutter_table["priority"] == 0
        is_p0 |= np.isin(self.stab.shutter_table["id"], force_p0_ids)
        
        is_p1 = (~is_p0) & (self.stab.shutter_table["priority"] == 1)
        is_p12 = (~is_p0) & (self.stab.shutter_table["priority"] < 3)
        is_p3 = (~is_p0) & (self.stab.shutter_table["priority"] <= 3.99)

        olap_p0 = ((self.grating_olap * is_p0).sum(axis=1) > 0) & (~is_p0)
        olap_p3 = ((self.grating_olap * (is_p0 | is_p3)).sum(axis=1) > 0)
        olap_p3 &= ~(is_p0 | is_p3)
        
        clean_prism = self.stab.ok_prism & ~(olap_p3)
        if not allow_filler_stuck_open:
            clean_prism &= ~(self.stuck_olap & ~(is_p0 | is_p3))
        
        return clean_prism

    def update_shutter_table_mask(self, allocated, which="prism"):
        """
        Update flags in shutter table for prism / grating
        """
        if not hasattr(self.stab, "ok_prism_orig"):
            self.stab.ok_prism_orig = self.stab.ok_prism & True
            self.stab.ok_grating_orig = self.stab.ok_grating & True
        
        if which == "prism":
            self.stab.ok_prism = allocated["clean_mask"] & True
        else:
            self.stab.ok_grating = allocated["clean_mask"] & True

    def plot_full_msa(self, allocated, figsize=(10, 10), max_priority=7, **kwargs):
        """
        Make a figure of the source locations in the MSA coordinates
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sub = self.stab.shutter_table["priority"] < max_priority

        ypad = allocated["ypad"]

        for j in np.where(sub)[0]:
            row = self.stab.shutter_table[j]

            pri = int(np.floor(row["priority"]))
            if pri in [0, 1, 2, 3]:
                color = PRIORITY_COLOR[pri]
                xpad = allocated["xpad"]
            else:
                color = "0.8"
                xpad = allocated["filler_xpad"]

            fc = color if allocated["clean_mask"][j] else "None"
            alpha = 0.5 if allocated["clean_mask"][j] else 0.2
            zorder = (100 - pri) + 0.2 * allocated["clean_mask"][j]

            if allocated["clean_mask"][j]:

                ax.scatter(
                    self.xquad[j],
                    self.yquad[j],
                    marker=".",
                    c=color,
                    zorder=zorder + 0.2,
                )

                path = Path(
                    np.array(
                        [
                            self.xquad[j] + xpad * np.array([-1, 1, 1, -1, -1]),
                            self.yquad[j] + ypad * np.array([-1, -1, 1, 1, -1]),
                        ]
                    ).T
                )

                _ = ax.add_patch(
                    PathPatch(path, fc=fc, ec=color, alpha=alpha, zorder=zorder)
                )

            else:
                ax.scatter(
                    self.xquad[j],
                    self.yquad[j],
                    marker=".",
                    fc="None",
                    ec=color,
                    zorder=zorder + 0.2,
                )

        ax.grid()

        xlim = (377 + self.dxcen + 20, -20)
        ylim = (171 + self.dycen + 10, -10)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plan_utils.set_axis_ticks(ax)

        return fig

    def plot_traces(self, allocated, att=None, show_all=False, priorities=None):
        """
        Make a plot showing the full traces
        """
        fig = self.stab.plot_slitlets(att=att, draw_polygons=False)

        in_mask = allocated["clean_mask"] & True
        if priorities is not None:
            in_mask &= np.isin(self.stab.shutter_table["priority"], priorities)

        for i, det in enumerate(["NRS1", "NRS2"]):
            ax = fig.axes[i]

            for j, in_mask_j in enumerate(in_mask):
                if (not in_mask_j) & (not show_all):
                    continue

                row = self.stab.shutter_table[j]
                pri = int(np.floor(row["priority"]))
                if priorities is not None:
                    if pri not in priorities:
                        continue

                path = self.grating_paths[j][i]
                if path is None:
                    continue

                if pri in [0, 1, 2, 3]:
                    color = PRIORITY_COLOR[pri]
                    alpha = 0.5
                else:
                    color = "0.8"
                    alpha = 0.5

                zorder = (100 - pri)
                if in_mask_j:
                    fc = color
                else:
                    fc = "None"
                    zorder -= 0.5

                _ = ax.add_patch(
                    PathPatch(
                        path,
                        fc=fc,
                        ec=color,
                        alpha=alpha,
                        zorder=zorder
                    )
                )

            # for j in range(No):
            for pj in self.so_grating_paths:
                path = pj[i]
                if path is None:
                    continue

                color = "purple"
                _ = ax.add_patch(
                    PathPatch(path, fc=color, ec=color, alpha=0.5, zorder=10000)
                )

        if self.grating != "PRISM":
            for ax in fig.axes:
                ax.set_xlim(0, 2048)
                ax.set_xticks(range(0, 2049, 512))

        return fig

        # fig.savefig(f'{self.pointing_name}_{grating}_overlaps.png')

    def write_outputs(self, allocated=None, att=None, center_threshold=1.0, show_all=False, **kwargs):
    
        if allocated is None:
            allocated = {
                "clean_mask": (
                    self.stab.ok_prism if self.grating == "prism"
                    else self.stab.ok_grating
                ),
                "ypad": 5,
                "xpad": 7,
                "filler_xpad": 7,
            }

        fig = self.plot_full_msa(allocated)
        fig.savefig(f'{self.stab.pointing_name}_{self.grating}_msa.png'.lower())
    
        fig = self.plot_traces(allocated, att=att, show_all=show_all)
        fig.savefig(f'{self.stab.pointing_name}_{self.grating}.png'.lower())

        _ = self.stab.write_outputs(
            att=att,
            make_figures=False,
            grating=self.grating.lower(),
            filter=self.filter.lower(),
            center_threshold=center_threshold,
            **kwargs
        )