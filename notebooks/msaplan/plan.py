import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from scipy.spatial import ConvexHull

import astropy.units as u
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

import pysiaf

from grizli import utils

from . import utils as plan_utils
from . import shutters

VERBOSITY = True

DISP_SLIT_STEP = 0.27
SPAT_SLIT_STEP = 0.53

def compute_apa_offset(
    cat,
    cat_coord=None,
    apt_coord="02:17:29.8978 -05:15:31.19",
    plan_apa=202.9918,
    swap_coordinates=True,
    verbose=False,
):
    """
    Eq. 10 from Bonaventura et al. for the small PA offset based on the median catalog position

    Seems to have a sign error relative to what APT calculates internally, used if `swap_coordinates=True`

    """
    if cat_coord is None:
        cat_ra = np.median(cat["ra"])
        cat_dec = np.median(cat["dec"])
        cat_coord = SkyCoord(cat_ra, cat_dec, unit="deg")

    ac = cat_coord.ra.deg / 180 * np.pi
    dc = cat_coord.dec.deg / 180 * np.pi

    if hasattr(apt_coord, "upper"):
        apt_coord = SkyCoord(apt_coord, unit=("hour", "deg"))

    dx = apt_coord.spherical_offsets_to(cat_coord)

    ap = apt_coord.ra.deg / 180 * np.pi
    dp = apt_coord.dec.deg / 180 * np.pi

    # Needs to swap to agree with APT
    if swap_coordinates:
        cx, cy = ac * 1, dc * 1
        ac, dc = ap, dp
        ap, dp = cx, cy

    num = np.sin(ap - ac) * (np.sin(dc) + np.sin(dp))
    den = np.cos(dc) * np.cos(dp) + np.cos(ap - ac) * (1 + np.sin(dc) * np.sin(dp))

    dphi = np.arctan(num / den) / np.pi * 180

    msg = f"Catalog offset: {dx[0].to(u.arcsec):5.2f} {dx[1].to(u.arcsec):5.2f} new APA: {plan_apa + dphi:.5f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return plan_apa + dphi, dphi


def rotate_coords(x, y, deg):
    """Rotate coordinates about the cartesian origin by `deg` degrees"""

    rad = deg / 180.0 * np.pi

    _mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    xr, yr = np.array([x, y]).T.dot(_mat).T

    return xr, yr


class MaskPlanAPT:

    SHUTTER_COLUMN_ANGLE = 0.731181  # Angle between cross-dispersion and V3

    def __init__(
        self,
        plan_file="uds-south-1-v4-apt.json",
        msa_target_csv="4233-obs7-exp1-c1e1n1-PRISM-CLEAR.csv",
        coeffs_key="prism",
        APER_NAME="NRS_FULL_MSA",
        fit_va=True,
        **kwargs,
    ):
        """ """
        import pysiaf

        self.plan_file = plan_file
        self.msa_target_csv = msa_target_csv
        self.coeffs_key = coeffs_key

        self.targ = self.read_msa_targets()

        self.ap = pysiaf.Siaf("NIRSPEC")[APER_NAME]

        self.V3IdlYAngle = self.ap.V3IdlYAngle * 1
        self.V2Ref = self.ap.V2Ref * 1
        self.V3Ref = self.ap.V3Ref * 1

        self.plan = self.read_plan()

        self.apt_coord = SkyCoord(
            self.plan["referencePointing"]["ra"],
            self.plan["referencePointing"]["dec"],
            unit="deg",
        )

        self.ap0 = self.ref_aper()

        if fit_va:
            self.fit_velocity_aberration()
        else:
            self.va_aper = self.ap0

        self.set_v2_v3(**kwargs)

    def __repr__(self):
        return (
            f"MaskPlanAPT(plan_file='{self.plan_file}',"
            + f" msa_target_csv='{self.msa_target_csv})"
        )

    def read_plan(self):
        with open(self.plan_file) as fp:
            plan = json.load(fp)
        return plan

    def read_msa_targets(self, source_types=["Primary", "Filler"]):
        targ = utils.read_catalog(self.msa_target_csv)

        prim = np.isin(targ["Source Type"], source_types)
        msg = (
            f"{self.msa_target_csv}: {prim.sum()} / {len(targ)} "
            + f'{" + ".join(source_types)} sources'
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        targ = targ[prim]

        # Rename some columns
        targ["ra"] = targ["Source RA (Degrees)"]
        targ["dec"] = targ["Source Dec (Degrees)"]
        targ["row"] = targ["Row (Spat)"]
        targ["col"] = targ["Column (Disp)"]
        targ["dx"] = targ["Offset (x)"]
        targ["dy"] = targ["Offset (y)"]

        targ["xrow"] = targ["row"] + targ["dy"] * 1.0
        targ["xcol"] = targ["col"] + targ["dx"] * 1.0

        targ["coo"] = SkyCoord(targ["ra"], targ["dec"], unit="deg")

        return targ

    def set_v2_v3(self, calculate_v2v3=True, **kwargs):
        # ap = self.ref_aper
        ap = self.va_aper

        if calculate_v2v3:
            self.targ["v2"], self.targ["v3"] = ap.sky_to_tel(
                self.targ["ra"], self.targ["dec"]
            )
            self.targ.meta["v2v3src"] = "Calculated"
        else:
            print("Use tabulated V2, V3")
            self.targ["v2"], self.targ["v3"] = (
                self.targ["V2 (arcsec)"] * 1,
                self.targ["V3 (arcsec)"] * 1,
            )
            self.targ.meta["v2v3src"] = "From CSV"

    def plan_reference_coords(self, idx=0):
        ra_ref = self.plan["configs"][0]["exposures"][idx]["ra"]
        dec_ref = self.plan["configs"][0]["exposures"][idx]["dec"]

        # ra_ref = self.targ['Fiducial RA (Degrees)'][0]
        # dec_ref = self.targ['Fiducial Dec (Degrees)'][0]

        ref_coo = SkyCoord(ra_ref, dec_ref, unit="deg")
        return ref_coo

    def get_aperture(
        self, ref_coo, pa_offset=0.0, fiducial=False, with_va=False, **kwargs
    ):
        """
        Get a SIAF aperture for a new pointing relative to the plan
        """
        from pysiaf.utils import rotations

        ap = self.ap

        if fiducial:
            APA = self.targ["Aperture PA (Degrees)"][0]
        else:
            APA, dphi = compute_apa_offset(
                None,
                cat_coord=ref_coo,
                apt_coord=self.apt_coord,
                plan_apa=self.plan["aperturePA"],
                swap_coordinates=True,
            )

        PA_V3 = (APA - self.V3IdlYAngle) + pa_offset

        self.att = rotations.attitude(
            self.V2Ref, self.V3Ref, ref_coo.ra.deg, ref_coo.dec.deg, PA_V3
        )

        ap.set_attitude_matrix(self.att)

        if with_va & self.has_va:
            ap = ApertureWithVA(
                ap,
                scale_v2=self.va_scale_v2,
                scale_v3=self.va_scale_v3,
            )

        return ap

    def ref_aper(self):
        ref_coo = self.plan_reference_coords(idx=0)
        ap = self.get_aperture(ref_coo, fiducial=True)
        return ap

    @property
    def has_va(self):

        try:
            va_max = np.max(np.abs([self.va_scale_v2, self.va_scale_v3]) - 1.0)
            return va_max > 1.0e-6

        except AttributeError:
            return False

    def fit_velocity_aberration(self, **kwargs):
        from scipy.optimize import minimize

        va_aper = ApertureWithVA(self.ap0)
        x0 = np.array([100.0])

        res = minimize(
            objfun_va_aper,
            x0=x0,
            args=(va_aper, self.targ),
            method="powell",
            tol=1.0e-4,
        )
        y0 = objfun_va_aper(x0, va_aper, self.targ)
        y1 = objfun_va_aper(res.x, va_aper, self.targ)

        msg = f"    fit_velocity_aberration: {res.x[0]/100.:.6f}  ({y0:6.3f} -> {y1:6.3f})"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        self.va_scale_v2 = self.va_scale_v3 = res.x[0] / 100.0
        self.va_aper = va_aper


class ApertureWithVA:

    def __init__(self, ap, scale_v2=1.0, scale_v3=1.0, **kwargs):
        self.ap = ap
        self.scale_v2 = scale_v2
        self.scale_v3 = scale_v3

    def sky_to_tel(self, ra, dec):
        v2, v3 = self.ap.sky_to_tel(ra, dec)

        v2 = (v2 - self.ap.V2Ref) * self.scale_v2 + self.ap.V2Ref
        v3 = (v3 - self.ap.V3Ref) * self.scale_v3 + self.ap.V3Ref

        return v2, v3

    def tel_to_sky(self, v2i, v3i):
        v2 = (v2i - self.ap.V2Ref) / self.scale_v2 + self.ap.V2Ref
        v3 = (v3i - self.ap.V3Ref) / self.scale_v3 + self.ap.V3Ref

        ra, dec = self.ap.tel_to_sky(v2, v3)
        return ra, dec


def objfun_va_aper(theta, va_ap, targ):

    if len(theta) == 1:
        scale_v2 = theta[0] / 100.0
        scale_v3 = scale_v2

    elif len(theta) == 2:
        scale_v2 = theta[0] / 100.0
        scale_v3 = theta[1] / 100.0

    va_ap.scale_v2 = scale_v2
    va_ap.scale_v3 = scale_v3

    # offset_coo = ref_coo.directional_offset_by(offset_ang, offset_size)
    # ap = self.get_aperture(offset_coo, fiducial=True, pa_offset=pa_offset)
    v2, v3 = va_ap.sky_to_tel(targ["ra"], targ["dec"])

    dv2 = v2 - targ["V2 (arcsec)"]
    dv3 = v3 - targ["V3 (arcsec)"]

    dr = np.sqrt(dv2**2 + dv3**2)
    loss = dr.sum()

    return loss


def fit_msa_transformation(targ, fit_degree=4):
    """
    Fit for the transform between v2,v3 and MSA shutter coordinates
    """
    p2 = Polynomial2D(degree=fit_degree)
    fitter = LinearLSQFitter()

    coeffs = {}
    inv_coeffs = {}

    for qi in np.unique(targ["Quadrant"]):

        q = targ["Quadrant"] == qi

        pv2 = fitter(p2, targ["xrow"][q], targ["xcol"][q], targ["v2"][q])
        pv3 = fitter(p2, targ["xrow"][q], targ["xcol"][q], targ["v3"][q])
        coeffs[qi] = pv2, pv3

        prow = fitter(p2, targ["v2"][q], targ["v3"][q], targ["xrow"][q])
        pcol = fitter(p2, targ["v2"][q], targ["v3"][q], targ["xcol"][q])
        inv_coeffs[qi] = prow, pcol

    return coeffs, inv_coeffs


def check_transforms(msa, coeffs, inv_coeffs, dx=0.02):

    # ap = msa.ref_aper
    ap = msa.va_aper

    targ = msa.targ

    if targ.meta["v2v3src"] == "Calculated":
        v2, v3 = ap.sky_to_tel(targ["ra"], targ["dec"])
    else:
        v2, v3 = targ["V2 (arcsec)"] * 1, targ["V3 (arcsec)"] * 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Check
    for qi in np.unique(targ["Quadrant"]):
        q = targ["Quadrant"] == qi
        prow, pcol = inv_coeffs[qi]

        axes[0].scatter(
            targ["row"][q],
            (pcol(v2, v3) - targ["xcol"])[q],
            c=targ["Quadrant"][q],
            vmin=0,
            vmax=4,
        )

        axes[0].scatter(
            targ["row"][q],
            (prow(v2, v3) - targ["xrow"])[q] + 0.02,
            c=targ["Quadrant"][q],
            vmin=0,
            vmax=4,
        )

        axes[1].scatter(
            (prow(v2, v3) - targ["xrow"])[q],
            (pcol(v2, v3) - targ["xcol"])[q],
            c=targ["Quadrant"][q],
            vmin=0,
            vmax=4,
        )

    axes[0].set_ylim(-0.5 * dx, 1.5 * dx)
    axes[0].set_xlabel("MSA row")
    axes[0].set_ylabel("Shutter residual + offset")

    axes[1].set_xlim(-dx, dx)
    axes[1].set_ylim(-dx, dx)
    axes[1].set_xlabel("row offset")
    axes[1].set_ylabel("col offset")

    for ax in axes:
        ax.grid()

    fig.tight_layout(pad=1)


OFFSET_TO_PA = 0.0


class Planner:

    def __init__(self, cat, mplan, coeffs_key="prism", smask=None, **kwargs):
        """
        Allocate shutters based on priorities, etc.
        """

        self.cat = cat
        # if 'ix' not in self.cat.colnames:
        self.cat["ix"] = np.arange(len(cat), dtype=int)

        self.mplan = mplan

        self.set_coeffs(coeffs_key=coeffs_key)

        if smask is None:
            self.smask = shutters.ShutterMask()
        else:
            self.smask = smask

    def set_coeffs(self, coeffs_key="prism"):

        _ = plan_utils.load_slit_transformation_coeffs()

        all_coeffs, all_inv_coeffs = _

        self.coeffs = all_coeffs[coeffs_key]
        self.inv_coeffs = all_inv_coeffs[coeffs_key]
        self.coeffs_key = coeffs_key

    def get_shutters(
        self,
        selection=None,
        disp_offset=0.0,
        spat_offset=0.0,
        pointing_center=None,
        **kwargs,
    ):
        """
        Get shutters for a particular pointing
        """

        # Pointing
        ref_coo = self.mplan.plan_reference_coords()
        if pointing_center is None:
            pa = -(3 * np.pi / 2 + np.arctan2(spat_offset, disp_offset)) * u.radian
            pa += (
                (self.mplan.plan["aperturePA"] + OFFSET_TO_PA)
                / 180.0
                * np.pi
                * u.radian
            )

            off = np.sqrt(disp_offset**2 + spat_offset**2) * u.arcsec

            offset_coo = ref_coo.directional_offset_by(pa, off)
        else:
            offset_coo = pointing_center

        offset_dx = ref_coo.spherical_offsets_to(offset_coo)

        if selection is not None:
            shut = self.cat["ID", "ix", "RA", "Dec", "Priority", "Weight"][selection]
        else:
            shut = self.cat["ID", "ix", "RA", "Dec", "Priority", "Weight"]

        ap = self.mplan.get_aperture(offset_coo, with_va=True)
        self.offset_ap = ap

        v2, v3 = ap.sky_to_tel(shut["RA"], shut["Dec"])

        shut["quad"] = 0
        shut["xrow"] = 0.0
        shut["xcol"] = 0.0

        for qi in [1, 2, 3, 4]:
            prow, pcol = self.inv_coeffs[qi]
            row = prow(v2, v3)
            col = pcol(v2, v3)

            # if qi == 1:
            #     in_quad = (row > 15) & (row < 169) & (col > 10) & (col < 365)
            # elif qi == 2:
            #     in_quad = (row > 3) & (row < 169) & (col > 10) & (col < 365)
            # elif qi == 3:
            #     in_quad = (row > 15) & (row < 169) & (col > 1) & (col < 365)
            # elif qi == 4:
            #     in_quad = (row > 3) & (row < 169) & (col > 1) & (col < 365)

            # From APT table
            if qi == 1:
                in_quad = (row >= 14) & (row <= 170) & (col >= 10) & (col <= 365)
            elif qi == 2:
                in_quad = (row >= 7) & (row <= 158) & (col >= 10) & (col <= 365)
            elif qi == 3:
                in_quad = (row >= 14) & (row <= 167) & (col >= 2) & (col <= 355)
            elif qi == 4:
                in_quad = (row >= 2) & (row <= 157) & (col >= 3) & (col <= 359)

            shut["quad"][in_quad] = qi
            shut["xrow"][in_quad] = row[in_quad]
            shut["xcol"][in_quad] = col[in_quad]

        shut.meta["ra_ref"] = offset_coo.ra.deg, "Pointing center RA"
        shut.meta["dec_ref"] = offset_coo.dec.deg, "Pointing center Dec."
        shut.meta["offset0"] = offset_dx[0].value
        shut.meta["offset1"] = offset_dx[1].value

        return shut[shut["quad"] > 0]

    def make_mask(
        self,
        root="rubies",
        disp_offset=0.0,
        spat_offset=0.0,
        pointing_center=None,
        faint_limit=None,
        max_priority=None,
        output_type="full",
        valid_shutters=True,
        cscale=0.8,
        simple_mask=False,
        simple_shutter_pad=148,  # for prism
        verbose=True,
        only_shutters=False,
    ):
        """ """
        # Slitlet centering tolerance
        pw = (0.07 / 0.27) * 0.3
        ph = (0.07 / 0.53) * 0.3

        pw *= cscale
        ph *= cscale

        debug = verbose > 1

        # Setup
        key = f"{spat_offset:03.0f}_{disp_offset:03.0f}".replace("-", "m")
        name = f"{root}_{key}_prism"

        # Raw shutters
        selection = None
        if faint_limit is not None:
            selection = self.cat["Magnitude"] < faint_limit

        if max_priority is not None:
            pi = self.cat["Priority"] < max_priority
            if selection is None:
                selection = pi
            else:
                selection &= pi

        if verbose & (selection is not None):
            print(f"Initial selection: {selection.sum()}")

        shut = self.get_shutters(
            selection=selection,
            disp_offset=disp_offset,
            spat_offset=spat_offset,
            pointing_center=pointing_center,
        )

        self.cat["quad"] = 0
        self.cat["quad"][shut["ix"]] = shut["quad"]

        # Just calculate shutters within quadrants
        if output_type == "quad":
            return shut

        row = shut["xrow"]
        col = shut["xcol"]
        sample = shut["ix"] > -1

        if verbose:
            print("In quadrants: ", sample.sum())

        # Apply shutter centering
        dr = np.abs(row - np.floor(row))
        dc = np.abs(col - np.floor(col))
        sample &= (dc > pw) & (dc < 1 - pw) & (dr > ph) & (dr < 1 - ph)

        if 1:
            grow = 1.5
            # sample &= (dc > pw*grow) & (dc < 1-pw*grow) & (dr > ph*3) & (dr < 1-ph*grow)
            sample &= (
                (dc > pw * grow)
                & (dc < 1 - pw * grow)
                & (dr > ph * grow)
                & (dr < 1 - ph * grow)
            )  # Oct 2024

        if output_type == "center":
            return shut[sample]

        if debug:
            print("xx center: ", (sample).sum())

        rowi = np.cast[int](np.floor(row)) - 1
        coli = np.cast[int](np.floor(col))

        if 0:
            slits = [f"{qi} {c} {r}" for qi, r, c in zip(shut["quad"], rowi, coli)]
            bad_slits = np.in1d(slits, self.smask.BAD_SHUTTERS)
        else:
            slits = np.cast[int](1e6 * shut["quad"] + 1000 * coli + rowi).tolist()
            bad_slits = np.in1d(slits, self.smask.BAD_SHUTTERS_IX)

        # 3-shutter slitlets
        if 1:
            # This version is used for existing masks
            for roff in [
                2,
                1,
                0,
            ]:
                for coff in [0]:  # [1,0,-1]:
                    # xslits = [f'{qi} {c} {r}' for qi, r, c in zip(shut['quad'], rowi+roff, coli+coff)]
                    xslits = np.cast[int](
                        1e6 * shut["quad"] + 1000 * (coli + coff) + (rowi + roff)
                    ).tolist()
                    bad_slits |= np.in1d(xslits, self.smask.BAD_SHUTTERS_IX)
        else:
            # Swap rows, columns
            for roff in [0]:
                for coff in [1, 0, -1]:  # [1,0,-1]:
                    # xslits = [f'{qi} {c} {r}' for qi, r, c in zip(shut['quad'], rowi+roff, coli+coff)]
                    xslits = np.cast[int](
                        1e6 * shut["quad"] + 1000 * (coli + coff) + (rowi + roff)
                    ).tolist()
                    bad_slits |= np.in1d(xslits, self.smask.BAD_SHUTTERS_IX)

        if valid_shutters:
            sample &= ~bad_slits

        if verbose:
            print(f"Centered slits: {sample.sum()}")

        if output_type == "valid":
            return shut[sample]

        all_rows = rowi[sample]
        all_cols = coli[sample]
        all_raw_rows = (row - 1)[sample]
        all_raw_cols = col[sample]

        so = np.argsort(all_cols)

        shutter_table = utils.GTable()
        shutter_table.meta["name"] = name
        shutter_table.meta["ra_off"] = disp_offset
        shutter_table.meta["dec_off"] = spat_offset
        shutter_table.meta["cscale"] = cscale

        for k in shut.meta:
            shutter_table.meta[k] = shut.meta[k]

        shutter_table.meta["parent"] = self.mplan.plan_file

        shutter_table["id"] = shut["ID"][sample]
        shutter_table["ix"] = shut["ix"][sample]
        shutter_table["shutter_quadrant"] = shut["quad"][sample]

        # Swapped in definition in APT & Pipeline !?
        shutter_table["shutter_row"] = all_cols
        shutter_table["shutter_column"] = all_rows

        shutter_table["raw_row"] = all_raw_cols
        shutter_table["raw_col"] = all_raw_rows

        if len(shutter_table) > 0:
            if simple_mask:
                keep = []

                so = np.argsort(self.cat["Weight"][shutter_table["ix"]])[::-1]

                for j in so:
                    if len(keep) == 0:
                        keep.append(j)
                        continue

                    shutter_row = shutter_table[j]
                    same_q = (
                        shutter_table["shutter_quadrant"][keep]
                        == shutter_row["shutter_quadrant"]
                    )
                    if same_q.sum() == 0:
                        keep.append(j)
                        continue

                    dc = (
                        shutter_row["shutter_column"]
                        - shutter_table["shutter_column"][keep][same_q]
                    )
                    if np.min(np.abs(dc)) > 2:
                        keep.append(j)
                        continue

                    dr = (
                        shutter_row["shutter_row"]
                        - shutter_table["shutter_row"][keep][same_q]
                    )
                    if np.min(np.abs(dr[np.abs(dc) <= 2])) > simple_shutter_pad:
                        keep.append(j)

                if verbose:
                    print(
                        f"simple_mask: keep {len(keep)} of  {len(shutter_table)} slits"
                    )

                shutter_table = shutter_table[keep]

        return shutter_table


class MSAPointingOptimizer:
    def __init__(
        self,
        mplan=None,
        source_table=None,
        oversample_shutters=5,
        weight_column="WeightNorm",
        smask=None,
        valid_key="all",
        **kwargs,
    ):

        import pysiaf

        self.weight_column = weight_column
        self.oversample_shutters = oversample_shutters

        if smask is None:
            self.smask = shutters.ShutterMask()
        else:
            self.smask = smask

        self.set_shutter_mask(valid_key=valid_key)

        # siaf_nrs = pysiaf.Siaf("NIRSpec")
        # aper_msa = siaf_nrs["NRS_FULL_MSA"]
        #
        # self.msa_footprint = utils.SRegion(
        #     np.array(aper_msa.corners("tel")), wrap=False
        # )
        #
        # self.msa_footprint_path = self.msa_footprint.path[0]

        self.mplan = mplan
        self.msa_ref_coords = self.mplan.plan_reference_coords()
        self.msa_ref_v2 = self.mplan.ap0.V2Ref * 1
        self.msa_ref_v3 = self.mplan.ap0.V3Ref * 1

        self.set_source_table(source_table)


    def set_shutter_mask(self, valid_key="all"):
        
        # self.set_msa_footprint(valid_key=valid_key)
        self.msa_footprint = plan_utils.shutter_mask_footprint()[valid_key]
        
        self.msa_footprint_paths = [
            self.msa_footprint[quad].path[0] for quad in self.msa_footprint
        ]

        self.shutter_tab = self.smask.VALID_TAB[valid_key]
        self.shutter_tab["norm_nn"] = (
            self.shutter_tab["nn"] / self.shutter_tab["nn"].max()
        )
        self.shutter_tree = self.smask.VALID_SHUTTER_TREE[valid_key]
        self.valid_key = valid_key

    def set_source_table(self, source_table):

        self.source_table = source_table
        v2v3 = self.mplan.va_aper.sky_to_tel(
            self.source_table["RA"], self.source_table["Dec"]
        )
        self.targ_v2v3 = np.array([*v2v3]).T

        if self.weight_column in source_table.colnames:
            self.weight = source_table[self.weight_column]
        else:
            self.weight = np.ones(len(self.source_table))

    def offset_coords(self, disp_offset=0, spat_offset=0, simple=True):
        """ """
        pa = -(3 * np.pi / 2 + np.arctan2(spat_offset, disp_offset)) * u.radian
        pa += (self.mplan.plan["aperturePA"] + OFFSET_TO_PA) / 180.0 * np.pi * u.radian

        off = np.sqrt(disp_offset**2 + spat_offset**2) * u.arcsec

        offset_coo = self.msa_ref_coords.directional_offset_by(pa, off)

        if simple:
            v2o, v3o = self.mplan.va_aper.sky_to_tel(
                offset_coo.ra.degree, offset_coo.dec.degree
            )

            v2_offset = v2o - self.msa_ref_v2
            v3_offset = v3o - self.msa_ref_v3

            targ_offset = self.targ_v2v3 + np.array([-v2_offset, -v3_offset])

        else:
            # shut = self.cat['ID','ix','RA','Dec','Priority','Weight']
            ap = self.mplan.get_aperture(offset_coo, with_va=True)
            self.offset_ap = ap

            v2, v3 = ap.sky_to_tel(
                self.source_table["RA"], self.source_table["Dec"]
            )
            targ_offset = np.array([v2, v3]).T

        return targ_offset

    def evaluate_offset(
        self, disp_offset=0, spat_offset=0.0, simple=True, toler=0.11, tree=None, plot_kwargs=None, plot_alpha=0.5, **kwargs
    ):

        targ_offset = self.offset_coords(
            disp_offset=disp_offset, spat_offset=spat_offset, simple=simple
        )
        in_msa = np.zeros(len(targ_offset), dtype=bool)
        for path in self.msa_footprint_paths:
            in_msa |= path.contains_points(targ_offset)

        if tree is None:
            tq = self.shutter_tree.query(
                targ_offset[in_msa, :], workers=5, distance_upper_bound=30
            )
        else:
            tq = tree.query(
                targ_offset[in_msa, :], workers=5, distance_upper_bound=30
            )

        in_shutter = tq[0] <= toler
        weight = self.weight[in_msa][in_shutter].sum()

        result = {
            "targ_offset": targ_offset,
            "in_msa": in_msa,
            "in_shutter": in_shutter,
            "weight": weight,
            "mean_tq": np.mean(tq[0][in_shutter]),
            "tq": tq,
        }
        
        if plot_kwargs is not None:
            if "ax" in plot_kwargs:
                ax = plot_kwargs["ax"]
                fig = None
            else:
                fig = plan_utils.show_msa_layout(**plot_kwargs)
                ax = fig.axes[0]

            pshow_ = {}

            unp = utils.Unique(
                np.floor(self.source_table[in_msa]["Priority"]).astype(int),
                verbose=False
            )
            
            for i, row in enumerate(self.source_table[in_msa]):
                p_i = int(np.floor(row['Priority']))
                if p_i in plan_utils.PRIORITY_COLOR:
                    c_i = plan_utils.PRIORITY_COLOR[p_i]
                    zorder=1001 - p_i
                else:
                    c_i = plan_utils.PRIORITY_COLOR["other"]
                    zorder = -1
                
                kws = {"fc": c_i, "ec": c_i}

                if not in_shutter[i]:
                    kws["fc"] = "None"
                
                ax.scatter(
                    *targ_offset[in_msa][i],
                    marker='o',
                    s=np.maximum(50 / 1.5**p_i, 20),
                    alpha=plot_alpha,
                    zorder=zorder,
                    label=(
                        f"P{p_i}  (N={unp[p_i].sum()})"
                        if p_i not in pshow_ else None
                    ),
                    **kws
                )
                
                pshow_[p_i] = True

            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            miss = self.source_table[~in_msa]
            unp = utils.Unique(
                np.floor(miss["Priority"]).astype(int),
                verbose=False
            )
            for p_i in unp.values:
                if p_i in plan_utils.PRIORITY_COLOR:
                    c_i = plan_utils.PRIORITY_COLOR[p_i]
                    zorder=1001 - p_i
                else:
                    c_i = plan_utils.PRIORITY_COLOR["other"]
                    zorder = -1
                
                ax.scatter(
                    *targ_offset[~in_msa][unp[p_i]].T,
                    marker='.',
                    color=c_i,
                    alpha=plot_alpha / 2.0,
                    zorder=zorder,
                    label=(f"P{p_i}" if p_i not in pshow_ else None),
                )
                pshow_[p_i] = True

            if fig is not None:
                leg = ax.legend()
                ax.text(
                    0.03, 0.97,
                    f"dx={disp_offset:.2f}  dy={spat_offset:.2f}".replace(
                        "d", r"$\Delta$"
                    ),
                    ha="left", va="top",
                    fontsize=7,
                    transform=ax.transAxes
                )
            
            result["fig"] = fig

        return result #targ_offset, in_msa, in_shutter, weight, np.mean(tq[0][in_shutter]), tq

    def optimize_pointing(
        self,
        x0=[0.0, 0.0],
        box_size=[10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05],
        Nper=32,
        min_dist=0.1,
        use_weight="weight",
        offset_loss_scale=5,
        weight_loss_scale=1000,
        force_simple=True,
        random_seed=1,
    ):
        """ """
        from tqdm import tqdm

        rows = []

        min_size = min_dist * Nper / 2
        # boxes = [min_size * 2**s for s in steps]

        print(f"min_size: {min_size:.2f}")

        dx0 = x0[0] * 1.0
        dy0 = x0[1] * 1.0

        if random_seed is not None:
            np.random.seed(random_seed)

        rows = []
        for box_size in box_size:
            step_size = box_size / 2 / Nper

            dx = (
                dx0
                + (np.arange(-Nper // 2, Nper // 2 + 1) + np.random.normal() * 0.0)
                * step_size
            )
            dy = (
                dy0
                + (np.arange(-Nper // 2, Nper // 2 + 1) + np.random.normal() * 0.0)
                * step_size
            )

            toler = np.maximum(step_size * np.sqrt(2), min_dist)
            simple = (
                step_size * np.sqrt(2) > min_dist
            ) & force_simple  # (box_size > boxes[-2])

            # _tab, _tree = padded_shutter_offset_table(oversample_shutters=self.oversample_shutters, pad_size=toler, min_size=0.2)
            _tree = self.shutter_tree

            dx, dy = np.meshgrid(dx, dy)
            dx = dx.flatten()
            dy = dy.flatten()

            if random_seed is not None:
                dx = (np.random.rand(Nper**2) - 0.5) * box_size + dx0
                dy = (np.random.rand(Nper**2) - 0.5) * box_size + dy0

            for dxi, dyi in tqdm(zip(dx, dy)):
                # targ_offset, in_msa, in_shutter, weight, med_off, tq 
                res = self.evaluate_offset(
                    disp_offset=dxi,
                    spat_offset=dyi,
                    simple=simple,
                    toler=toler,
                    tree=_tree,
                )

                # offset_loss = (tq[0]*np.exp(-tq[0]**2/2/4/toler**2))[tq[0] < 2*toler].sum()/4/toler
                offset_loss = np.exp(
                    -res["tq"][0] ** 2 / 2 / (offset_loss_scale * toler) ** 2
                )

                # Strong loss at toler = edge of shutter
                ### TURNED OFF for testing centering
                if 0:
                    scale_resid = res["tq"][0] / toler
                    offset_loss = np.exp(-(scale_resid**2) / 2) * np.exp(
                        -(scale_resid**13) / 2
                    )

                weight_loss = (
                    offset_loss
                    * self.weight[res["in_msa"]]
                    / self.weight.max()
                    * weight_loss_scale
                )

                weight_loss = weight_loss.sum()  # np.log10(weight_loss.sum())

                nnix_ = res["tq"][1][res["in_shutter"]]
                shutter_nn = self.shutter_tab["norm_nn"][nnix_]

                rows.append(
                    [
                        dxi,
                        dyi,
                        step_size,
                        res["in_shutter"].sum(),
                        res["weight"],
                        weight_loss.sum(),
                        offset_loss.sum(),
                        shutter_nn.sum(),
                        toler,
                        simple,
                    ]
                )

            off = utils.GTable(
                names=[
                    "dx",
                    "dy",
                    "step_size",
                    "in_shutter",
                    "weight",
                    "weight_loss",
                    "offset_loss",
                    "shutter_nn",
                    "toler",
                    "simple",
                ],
                rows=rows,
            )

            last = (off["toler"] <= toler) & (off["simple"] == simple)

            if use_weight == "weight_loss_scale":
                off["stat"] = (
                    off["weight"]
                    / off["weight"][last].max()
                    * (off["offset_loss"] / off["offset_loss"][last].max()) ** 2
                )
            elif use_weight == "count_scale":
                off["stat"] = (
                    off["in_shutter"]
                    * (off["offset_loss"] / off["offset_loss"][last].max()) ** 2
                )
            else:
                off["stat"] = off[use_weight] * 1.0

            step_sub_ = off["step_size"] > 0.15
            off["stat"][step_sub_] += off["shutter_nn"][step_sub_]

            ### Include shutter centering when tolerance less than a fraction
            ### of the shutter size
            # if step_size*3 <= min_dist:
            #     test = last & (off['step_size']*3 <= min_dist)
            #     off['stat'][last] -= off['mean_offset'][last]
            # else:
            test = last & True

            best = np.where(test)[0][np.argmax(off["stat"][test])]

            dx0 = off["dx"][best]
            dy0 = off["dy"][best]

            best_stat = off["stat"][best]
            best_n = off["in_shutter"][best]

            msg = (
                f"{box_size:>9.2f}   {step_size:>6.4f}  tol={toler:3.2f}"
                f"  dx={dx0:6.3f} dy={dy0:6.3f}  N={best_n} {best_stat:.2f}"
                f" (simple={simple})"
            )
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        off["last"] = last
        off["test"] = test

        off.meta["Nper"] = Nper
        off.meta["min_dist"] = min_dist
        off.meta["best"] = best
        off.meta["random_seed"] = random_seed

        return off, last, best


def plot_offset_table(off, cmap="gist_earth"):
    from scipy.spatial import ConvexHull

    steps = utils.Unique(off["step_size"], verbose=False)
    xsize = 4
    best = off.meta["best"]

    fig, axes = plt.subplots(1, steps.N, figsize=(xsize * steps.N, xsize))
    if steps.N == 1:
        axes = [axes]

    for i, step in enumerate(steps.values[::-1]):
        ax = axes[i]
        nper = np.sqrt(steps[step].sum())

        ax.hexbin(
            off["dx"][steps[step]],
            off["dy"][steps[step]],
            off["stat"][steps[step]],
            gridsize=int(
                np.minimum(56, int(nper * 0.9))
                * 0.7 ** (off.meta["random_seed"] is not None)
            ),
            reduce_C_function=np.max,
            cmap=cmap,
        )

        ax.text(
            0.05,
            0.05,
            f"Step: {step*1000:.0f} mas",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox={"fc": "w", "ec": "None", "alpha": 0.8},
        )

        if i < steps.N - 1:
            next = steps.values[::-1][i + 1]
            arr = np.array([off["dx"][steps[next]], off["dy"][steps[next]]]).T
            hull = ConvexHull(arr)
            ax.plot(
                *arr[np.append(hull.vertices, hull.vertices[:1]), :].T, color="pink"
            )

        ax.set_aspect("auto")

        ax.scatter(
            off["dx"][best], off["dy"][best], marker="o", fc="None", s=120, ec="magenta"
        )

        ax.set_xlabel(r"$\Delta$col, arcsec")
        ax.grid()
        ax.set_aspect(1)

    axes[0].set_ylabel(r"$\Delta$row, arcsec")

    fig.tight_layout(pad=0.5)

    return fig


# opt = MSAPointingOptimizer(msa=mplan, source_table=shut, oversample_shutters=5)

# plt.scatter(*opt.targ_v2v3.T, color='0.5')
# opt.msa_footprint.add_patch_to_axis(plt.gca(), fc='tomato', ec='None', alpha=0.2)

# targ_offset, in_msa, in_shutter, weight, med_offset, tq = opt.evaluate_offset()
# plt.scatter(*opt.targ_v2v3[in_msa,:][in_shutter,:].T, color='magenta')

# # off16, last, best = opt.optimize_pointing(Nper=16, box_size=[5,2,1,0.5,0.2,0.1], min_dist=0.8*0.1)
# # # off32, last, best = opt.optimize_pointing(Nper=32, steps=[3,2,1,0,-1,-2,-3], min_dist=0.8*0.1)
# # # off, last, best = opt.optimize_pointing(Nper=32, steps=[5,4,3,2,1,0,-1,-2,-3]) #,-1,-2,-3])

# # _ = plot_offset_table(off16)
# # _ = plot_offset_table(off32)

# # plt.xlim(270, 300)
# # plt.ylim(-450, -420)

def optimize_pointing(opt, disp_offset=0.0, spat_offset=0.0, large_start=2, optimize_type="large", N=64, random_seed=None, make_figures=True, verbose=VERBOSITY):
    """
    """

    info = f", N={N}  disp_offset={disp_offset:<7.3f} spat_offset={spat_offset:<7.3f}"

    if optimize_type == "large":
    
        msg = "Large search" + info
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        boxes = [N*2*4, N*2*1, N*2*0.2, N*2*0.05, N*2*0.0025][large_start:]

        off, last, best = opt.optimize_pointing(
            x0=[disp_offset, spat_offset],
            Nper=N,
            box_size=boxes,
            min_dist=0.11,
            use_weight='weight_loss_scale',
            weight_loss_scale=1e6 * 100,
            offset_loss_scale=1,
            random_seed=random_seed,
            force_simple=False
        )

    elif optimize_type == "count":

        msg = "Optimize count" + info
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        off, last, best = opt.optimize_pointing(
            x0=[disp_offset, spat_offset],
            Nper=N,
            box_size=[N*2*0.02, N*2*0.005],
            min_dist=0.11,
            use_weight='weight',
            weight_loss_scale=1,
            offset_loss_scale=1,
            random_seed=random_seed,
            force_simple=False
        )

    elif optimize_type == "fine":

        msg = 'Fine alignment' + info
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        off, last, best = opt.optimize_pointing(
            x0=[disp_offset, spat_offset],
            Nper=N,
            box_size=[N*2*0.005, N*2*0.001],
            min_dist=0.11,
            use_weight='weight_loss_scale',
            weight_loss_scale=1e2,
            offset_loss_scale=1e4,
            random_seed=random_seed,
            force_simple=False
        )
    
    else:
        # Just fine alignment
        msg = 'Fine #2' + info
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        off, last, best = opt.optimize_pointing(
            x0=[disp_offset, spat_offset],
            Nper=N,
            box_size=[N*2*0.02, N*2*0.005],
            min_dist=0.11,
            use_weight='weight_loss_scale',
            weight_loss_scale=1e6,
            offset_loss_scale=1,
            random_seed=random_seed,
            force_simple=False
        )
    
    best = off.meta['best']

    if make_figures:
        fig = plot_offset_table(off)
        
        eval_result = opt.evaluate_offset(
            disp_offset=off['dx'][best],
            spat_offset=off['dy'][best],
            simple=False, toler=0.11,
            tree=None,
            plot_kwargs={"ascale": 0.3},
        )

    result = {
        "disp_offset": off['dx'][best],
        "spat_offset": off['dy'][best],
        "offset": off,
        "best": best,
        "fig": fig,
        "eval": eval_result,
    }

    return result


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

        if ("mplan" in kwargs) & ("smask" in kwargs):
            self.set_optimizer(**kwargs)
        else:
            self.opt = None

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

    def set_optimizer(self, mplan=None, smask=None, valid_key="all", **kwargs):
        """
        """
        if (mplan is None) | (smask is None):
            self.opt = None
            return None

        self.opt = MSAPointingOptimizer(
            mplan=mplan,
            source_table=self.nrs,
            smask=smask,
            valid_key=valid_key,
            oversample_shutters=5,
            weight_column="Weight"
        )
        
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

        self.table = utils.GTable.read(
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

    def select_mag_ranges(self, selection=None, verbose=False):
        """
        """
        if selection is None:
            is_ref = self.nrs[self.columns["ref"]] > 0
        else:
            is_ref = selection & True

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

    def select_sources(self, mode="nrsrapid_f110w", scl=1.0, asinh=True, nper=20, page=0, output="display", selection=None, verbose=True, **kwargs):
        """
        """
        from IPython.display import display, Markdown

        cols = [self.columns[k] for k in ["id", "ra", "dec", "mag", "size"]]
        sub = self.nrs[cols][
            self.select_mag_ranges(selection=selection, verbose=False)[mode]
        ]

        sl = slice(page * nper, (page + 1) * nper)
        if verbose:
            print(
                f"{mode}  [{sl.start}:{sl.stop}] Ntot={len(sub)} "
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

    def plot_histogram(self, selection=None, **kwargs):
        """
        Make a figure
        """
        nrs = self.nrs

        if selection is None:
            is_ref = self.nrs[self.columns["ref"]] > 0
        else:
            is_ref = selection & True

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

    def check_pointing(self, disp_offset=0.0, spat_offset=0.0, **kwargs):
        """
        """
        result = self.opt.evaluate_offset(
            disp_offset=disp_offset,
            spat_offset=spat_offset,
            simple=False,
            tol=0.11
        )

        plot_kwargs = {
            "siaf_aper": self.opt.offset_ap,
            "show_footprints": "all",
            "legend": False,
        }

        fig = plan_utils.show_msa_layout(**plot_kwargs)
        ax = fig.axes[0]

        any_ref = self.nrs["Reference"] > 0

        tests = {}
        for i in range(0,4):
            ref_i = f"Reference{i}" if i > 0 else "Reference"
            if ref_i in self.nrs.colnames:
                tests[ref_i] = self.nrs[ref_i] > 0
                any_ref |= tests[ref_i]

                c_i = plt.cm.rainbow_r(i / 3.)
                in_msa_i = tests[ref_i] & result["in_msa"]
                not_in_msa_i = tests[ref_i] & ~result["in_msa"]

                ax.scatter(
                    self.nrs["RA"][in_msa_i],
                    self.nrs["Dec"][in_msa_i],
                    alpha=0.7, # - 0.1*i,
                    zorder=500 - i,
                    color=c_i,
                    marker="o",
                    s=40,
                    label=f"{ref_i} {in_msa_i.sum()}"
                )

                ax.scatter(
                    self.nrs["RA"][not_in_msa_i],
                    self.nrs["Dec"][not_in_msa_i],
                    alpha=(0.6 - 0.1*i) * 0.3,
                    zorder=500 - i,
                    fc="None",
                    ec=c_i
                )

        any_ref &= result["in_msa"]

        ax.text(
            0.03, 0.97,
            f"dx={disp_offset:.2f}  dy={spat_offset:.2f}".replace(
                "d", r"$\Delta$"
            ),
            ha="left", va="top",
            fontsize=7,
            transform=ax.transAxes
        )

        ax.legend(fontsize=7)

        result["fig"] = fig
        result["ref"] = tests
        result["any_ref"] = any_ref

        return result
