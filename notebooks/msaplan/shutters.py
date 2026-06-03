import os
import json

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from grizli import utils

from scipy.spatial import ConvexHull

import astropy.units as u
from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.coordinates import SkyCoord, Angle

from . import utils as plan_utils

VERBOSITY = True


def read_apt_shutter_csv(file):
    with open(file) as fp:
        lines = fp.readlines()[1:]

    data = np.array([np.cast[int](line.strip().split(",")) for line in lines])

    q = data * 0
    q[:365, :171] = 1
    q[365:, :171] = 3
    q[:365, 171:] = 2
    q[365:, 171:] = 4

    row, col = np.indices(data.shape)
    row = (row % 365) + 1
    col = (col % 171) + 1

    closed = data == 1

    tab = utils.GTable()
    tab["Q"] = q[closed]
    tab["x"] = tab["row"] = row[closed]
    tab["y"] = tab["col"] = col[closed]

    otab = utils.GTable()
    otab["Q"] = q[~closed]
    otab["x"] = otab["row"] = row[~closed]
    otab["y"] = otab["col"] = col[~closed]

    return tab, otab


def read_apt_shutter_csv_cube(file):
    """
    Read shutter CSV file as a [quadrant, row, col] cube
    """
    with open(file) as fp:
        lines = fp.readlines()[1:]

    data = np.array([np.cast[int](line.strip().split(",")) for line in lines])

    q = data * 0
    q[:365, :171] = 1
    q[365:, :171] = 3
    q[:365, 171:] = 2
    q[365:, 171:] = 4

    row, col = np.indices(data.shape)
    row = (row % 365) + 1
    col = (col % 171) + 1

    shutters = np.zeros((4, 365, 171), dtype=bool)

    for qi in [1, 2, 3, 4]:
        test = q == qi
        shutters[qi - 1, row[test] - 1, col[test] - 1] = data[test] > 0

    return shutters


def build_valid_shutter_table(
    oversample=4,
    offsets=[(1, 2), (-1, 2)],
    mask_prism=False,
    mask_g395m=False,
    coeffs=None,
    **kwargs,
):
    """ """
    shutter_cube = read_apt_shutter_csv_cube(
        plan_utils.data_path(file="all_open_may2026.csv")
    )

    if coeffs is None:
        _ = plan_utils.load_slit_transformation_coeffs()
        all_coeffs, all_inv_coeffs = _
        coeffs = all_coeffs["prism"]

    if mask_prism:
        shutter_cube |= read_apt_shutter_csv_cube(
            plan_utils.data_path(file="prism_gap_mask.csv")
        )
        shutter_cube |= read_apt_shutter_csv_cube(
            plan_utils.data_path(file="prism_red_mask.csv")
        )
        shutter_cube |= read_apt_shutter_csv_cube(
            plan_utils.data_path(file="prism_blue_mask.csv")
        )

    # For Complete G395M spectra
    if mask_g395m:
        shutter_cube |= read_apt_shutter_csv_cube(
            plan_utils.data_path(file="gap_red_blue_g395m.csv")
        )

    # valid for 3-shutter-nod
    invalid = shutter_cube & True
    for shift, axis in offsets:
        invalid |= np.roll(invalid, shift, axis=axis)

    quad, row, col = np.indices(invalid.shape) + 1
    tab = utils.GTable()

    # Oversample the shutter
    Ns = oversample
    steps = np.arange(-Ns, Ns + 1) / Ns * 0.25 * 0.46 / 0.53
    qx = [quad[~invalid]]
    rx = [row[~invalid]]
    cx = [col[~invalid]]

    if oversample > 0:
        for s in steps:
            qx.append(qx[0])
            rx.append(rx[0])
            cx.append(cx[0] + s)

    tab["Q"] = np.hstack(qx)
    tab["row"] = tab["x"] = np.hstack(rx) + 0.5
    tab["col"] = tab["y"] = np.hstack(cx) + 0.5

    tab["closed"] = False

    tab["v2"] = 0.0
    tab["v3"] = 0.0

    unq = utils.Unique(tab["Q"], verbose=False)
    for q in unq.values:
        tab["v2"][unq[q]] = coeffs[q][0](tab["y"][unq[q]], tab["x"][unq[q]])
        tab["v3"][unq[q]] = coeffs[q][1](tab["y"][unq[q]], tab["x"][unq[q]])

    return tab


def padded_shutter_offset_table(
    oversample_shutters=5, pad_size=0.2, min_size=0.2, **kwargs
):
    from scipy.spatial import cKDTree

    nx = int(pad_size / min_size)
    ny = nx // 2 + 1

    offsets = []
    for j in range(-ny, ny + 1):
        offsets.append((j, 2))

    for j in range(nx):
        if j == 0:
            pass
        for k in range(-j, j + 1):
            offsets.append((k, 1))

    tab = build_valid_shutter_table(
        oversample=oversample_shutters, offsets=offsets, **kwargs
    )

    print(f"shutter tree: {pad_size:.2f} {nx} {ny} N={len(tab)}")

    tel = np.array([tab["v2"], tab["v3"]]).T
    shutter_tree = cKDTree(tel)

    return tab, shutter_tree


class ShutterMask:
    def __init__(self, **kwargs):

        self.load_oper_mask(**kwargs)

        self.valid_shutters(**kwargs)

    def load_oper_mask(
        self, oper_versions=[17], all_open_file="all_open_may2026.csv", **kwargs
    ):
        """
        versions: [1,4,5,6,7,9,10,11,12,14,15,16,17]

        """
        # Operability, use all available files for a conservative mask

        CRDS_PATH = os.getenv("CRDS_PATH")

        BAD_SHUTTERS = []
        BAD_SHUTTERS_IX = []

        for ver in oper_versions:
            with open(
                os.path.join(
                    CRDS_PATH,
                    "references/jwst/nirspec",
                    f"jwst_nirspec_msaoper_{ver:04d}.json",
                )
            ) as fp:
                oper = json.load(fp)

            tab = utils.GTable(oper["msaoper"])
            ibad = tab["Vignetted"] == "yes"
            ibad |= tab["Internal state"] == "closed"

            BAD_SHUTTERS += [
                "{Q} {x} {y}".format(**row) for row in tab
            ]  # for q, x, y in zip(tab['Q'], tab['x'], tab['y'])]
            BAD_SHUTTERS_IX += np.cast[int](
                1e6 * tab["Q"] + 1000 * tab["x"] + tab["y"]
            ).tolist()

            Nix_ = len(np.unique(BAD_SHUTTERS_IX))
            msg = f"msaoper_{ver:04d} N={len(tab)}  Ntot={Nix_}"
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        open_shutters = tab["state"] == "open"
        open_shutters |= tab["Internal state"] == "open"

        stuck_open = ["{Q} {x} {y}".format(**row) for row in tab[open_shutters]]
        stuck_open_table = tab[open_shutters]

        file = plan_utils.data_path(file=all_open_file)
        tab, otab = read_apt_shutter_csv(file)

        BAD_SHUTTERS += ["{Q} {x} {y}".format(**row) for row in tab]
        BAD_SHUTTERS_IX += np.cast[int](
            1e6 * tab["Q"] + 1000 * tab["x"] + tab["y"]
        ).tolist()

        BAD_SHUTTERS = np.unique(BAD_SHUTTERS).tolist()
        BAD_SHUTTERS_IX = np.unique(BAD_SHUTTERS_IX).tolist()

        self.BAD_SHUTTERS = BAD_SHUTTERS
        self.BAD_SHUTTERS_IX = BAD_SHUTTERS_IX

        self.stuck_open = stuck_open
        self.stuck_open_table = stuck_open_table

    def valid_shutters(self, **kwargs):
        """
        Valid shutter lists, with masks for PRISM and G395M
        """
        from scipy.spatial import cKDTree

        VALID_TAB = {}
        VALID_SHUTTER_TREE = {}

        ##############
        # Complete G395M spectra
        msg = "Valid shutters: G395M"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        VALID_TAB["g395m"] = build_valid_shutter_table(
            oversample=5, offsets=[(1, 2), (-1, 2)], mask_g395m=True
        )
        tel = np.array([VALID_TAB["g395m"]["v2"], VALID_TAB["g395m"]["v3"]]).T

        VALID_SHUTTER_TREE["g395m"] = cKDTree(tel)
        neighbors = VALID_SHUTTER_TREE["g395m"].query_ball_tree(
            VALID_SHUTTER_TREE["g395m"], 0.5
        )
        VALID_TAB["g395m"]["nn"] = np.array([len(n) for n in neighbors])

        ##############
        # Complete PRISM spectra
        msg = "Valid shutters: PRISM"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        VALID_TAB["prism"] = build_valid_shutter_table(
            oversample=5, offsets=[(1, 2), (-1, 2)], mask_prism=True
        )
        tel = np.array([VALID_TAB["prism"]["v2"], VALID_TAB["prism"]["v3"]]).T

        VALID_SHUTTER_TREE["prism"] = cKDTree(tel)

        neighbors = VALID_SHUTTER_TREE["prism"].query_ball_tree(
            VALID_SHUTTER_TREE["prism"], 0.5
        )
        VALID_TAB["prism"]["nn"] = np.array([len(n) for n in neighbors])

        ##############
        # All valid shutters
        msg = "Valid shutters: All"
        utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY > 0))

        VALID_TAB["all"] = build_valid_shutter_table(
            oversample=5, offsets=[(1, 2), (-1, 2)]
        )
        tel = np.array([VALID_TAB["all"]["v2"], VALID_TAB["all"]["v3"]]).T

        VALID_SHUTTER_TREE["all"] = cKDTree(tel)

        # Neighbors
        neighbors = VALID_SHUTTER_TREE["all"].query_ball_tree(
            VALID_SHUTTER_TREE["all"], 0.5
        )
        VALID_TAB["all"]["nn"] = np.array([len(n) for n in neighbors])

        for k in VALID_TAB:
            stab = VALID_TAB[k]
            unq = utils.Unique(stab["Q"], verbose=False)

            off = np.zeros(2)

            for q in unq.values:
                sr = utils.catalog_bounding_polygon(
                    stab["v2"][unq[q]] - off[0],
                    stab["v3"][unq[q]] - off[1],
                    cosd=False,
                    buffer=(2, -1),  # (2,-1),
                    simplify=2,
                    scale=1.0,
                    wrap=False,
                )
                stab.meta[f"Q{q}"] = sr.polystr(precision=1)[0]

        if False:
            # mask put into data/shutter_mask.yml
            shutter_mask = {}
            labels = ["all", "g395m", "prism"]
            for k in VALID_TAB:
                stab = VALID_TAB[k]
                shutter_mask[labels[k]] = {}
                for q in range(1, 5):
                    shutter_mask[labels[k]][q] = stab.meta[f"Q{q}"]

            print(yaml.dump(shutter_mask))

        self.VALID_SHUTTER_TREE = VALID_SHUTTER_TREE
        self.VALID_TAB = VALID_TAB

    def valid_shutter_regions(self):
        """
        Valid shutter regions by quadrant.  Static version stored in
        ``data/shutter_mask.yml``.
        """
        shutter_mask = {}
        for k in self.VALID_TAB:
            stab = self.VALID_TAB[k]
            shutter_mask[k] = {}
            for q in range(1, 5):
                shutter_mask[k][q] = stab.meta[f"Q{q}"]

        return shutter_mask
