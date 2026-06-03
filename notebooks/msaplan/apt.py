import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from grizli import utils

from . import utils as plan_utils
from . import shutters, trace
from .utils import PRIORITY_COLOR


def make_apt_shutter_csv(shutter_table, slitlet=[2,1,0], name='test'):

    with open('all_closed.csv') as fp:
        lines = fp.readlines()[1:]

    data = np.array([np.cast[int](line.strip().split(',')) for line in lines])
    
    for s in shutter_table:
        qi = s['shutter_quadrant']
        if qi == 1:
            c0 = r0 = 0
        elif qi == 3:
            r0 = 365
            c0 = 0
        elif qi == 2:
            r0 = 0
            c0 = 171
        else:
            r0 = 365
            c0 = 171

        ri = s['shutter_row']
        ci = s['shutter_column']
        for roff in [2,1,0,]:
            data[ri+r0-1,ci+roff+c0-1] = 0
            
            #xslits = [f'{qi} {c} {r}' for qi, r, c in zip(shut['quad'], rowi+roff, coli+coff)]
    
    l = data.astype(str).tolist()
    print(name+'_shutters.csv')
    with open(name+'_shutters.csv','w') as fp:
        fp.write('# This CSV indicates which shutters should be open/closed on the MSA\n')
        for r in l:
            fp.write(','.join(r) + '\n')
        
    return data

def shutter_centering_offset(shutter_table, cscale=0.7, grow=1.5):
    pw = (0.07/0.27)*0.3
    ph = (0.07/0.53)*0.3

    pw *= cscale
    ph *= cscale
        
    # Apply shutter centering
    row = shutter_table["raw_col"]
    col = shutter_table["raw_row"]
    
    dr = np.abs(row-np.floor(row))
    dc = np.abs(col-np.floor(col))
    sample = (dc > pw) & (dc < 1-pw) & (dr > ph) & (dr < 1-ph)
        
    if grow is not None:
        sample &= (dc > pw*grow) & (dc < 1-pw*grow) & (dr > ph*grow) & (dr < 1-ph*grow)

    tab = utils.GTable()
    tab['dr'] = dr
    tab['dc'] = dc
    tab['valid'] = sample
    tab.meta["cscale"] = cscale
    tab.meta["grow"] = grow
    tab.meta["pw"] = pw
    tab.meta["ph"] = ph
    
    return tab

def show_shutter_centering(shutter_table, title=None, **kwargs):
    
    tab =  shutter_centering_offset(shutter_table, **kwargs)
    
    fig, ax = plt.subplots(1,1,figsize=(4,8))

    if "priority" in shutter_table.colnames:
        keep = shutter_table["priority"] != 11.

        for p in [0,1,2,3]:
            keep = shutter_table["priority"] == p
            
            ax.scatter(
                1 - tab["dc"][keep],
                1 - tab["dr"][keep],
                s=100,
                color=PRIORITY_COLOR[p], alpha=0.5, zorder=100-p,
            )
        # rest
        keep = (shutter_table["priority"] > 3.5) & (shutter_table["priority"] != 11)
        ax.scatter(
            1 - tab["dc"][keep],
            1 - tab["dr"][keep],
            color='0.7', alpha=0.3, zorder=10,
        )
        
    else:
        keep = np.ones(len(tab), dtype=bool)
        
        ax.scatter(
            1 - tab["dc"][keep],
            1 - tab["dr"][keep],
            c=tab["valid"][keep],
            vmin=0, vmax=1, cmap='RdYlGn', alpha=0.3
        )
    
    ax.set_xlim(0,1) #-0.1, 1.1)
    ax.set_ylim(0,1) #-0.05, 1.05)

    sr = utils.SRegion(np.array([[0,1,1,0,0], [0,0,1,1,0]]), wrap=False)
    sr.add_patch_to_axis(ax, ec='0.5', fc='pink', alpha=0.1, zorder=-11, label=f"cscale: {tab.meta['cscale']:.2f}")

    pw = tab.meta["pw"] * tab.meta["grow"]
    ph = tab.meta["ph"] * tab.meta["grow"]

    sr = utils.SRegion(np.array([[pw,1-pw,1-pw,pw,pw], [ph,ph,1-ph,1-ph,ph]]), wrap=False)
    sr.add_patch_to_axis(ax, ec='0.5', fc='0.99', alpha=1, zorder=-10)

    # ax.grid()

    ax.set_xticks([0,1]); ax.set_xticklabels([])
    ax.set_yticks([0,1]); ax.set_yticklabels([])
    ax.legend(loc='lower left')
    ax.set_title(title)
    fig.tight_layout(pad=0.5)
    
    return fig, tab

def make_plan_json(shutter_table, name='test', prism_sel=True, center_threshold=None, grating='g395m', filter='f290lp', keep_exposures=[0,1,2], **kwargs):

    import astropy.units as u
    from astropy.coordinates import Angle
    
    m = shutter_table.meta
    offset_dx = (Angle(m['offset0']*u.deg), Angle(m['offset1']*u.deg))
    
    with open(m['parent']) as fp:
        plan = json.load(fp)

    all_quads = shutter_table['shutter_quadrant'].tolist()
    all_ids = shutter_table['id'].tolist()
    all_cols = shutter_table['shutter_row'].tolist()
    all_rows = shutter_table['shutter_column'].tolist()

    so = np.argsort(all_cols)
        
    slitlets = [{'q':all_quads[i],
                 'd':int(all_cols[i]),
                 's':int(all_rows[i]),
                 'h':3 # 3-slitlet
                 } for i in so]

    sorted_ids = [all_ids[i] for i in so]
            
    plan['name'] = name
    plan['plannerSpecification']['planName'] = name

    sl = slice(0,len(sorted_ids))
    print(len(slitlets[sl]))

    plan['configs'][0]['slitlets'] = slitlets[sl]

    # Remove bkg
    if 'priority' in shutter_table.colnames:
        sky_rows = shutter_table['priority'] == 11
        if sky_rows.sum() > 0:
            print(f'Remove {sky_rows.sum()} sky sources from ID list')
            sky_ids = shutter_table['id'][sky_rows]
            is_sky = np.isin(sorted_ids, sky_ids)
            sorted_ids = np.array(sorted_ids)[~is_sky].tolist()

    if center_threshold is not None:
        fig, shut_center = show_shutter_centering(shutter_table, cscale=center_threshold, title=name)
        fig.savefig(f'{name}_centering.png')
        
        test = np.isin(sorted_ids, shutter_table["id"][shut_center["valid"]])
        print(f"Keep {test.sum()} / {len(test)} sources in ID list")
        sorted_ids = np.array(sorted_ids)[test].tolist()
        
    prism_exposures = []

    plan['catalog']['primariesName'] = plan['catalog']['name']
    plan['catalog']['fillersName'] = None
    
    for i in range(len(plan['configs'])):
        for ie in range(len(plan['configs'][i]['exposures'])):
            plan['configs'][i]['exposures'][ie]['sourceIds'] = sorted_ids

            ecoo = SkyCoord(plan['configs'][i]['exposures'][ie]['ra'],
                            plan['configs'][i]['exposures'][ie]['dec'],
                            unit='deg')
            
            eoff = ecoo.spherical_offsets_by(*offset_dx)
            plan['configs'][i]['exposures'][ie]['ra'] = eoff.ra.deg
            plan['configs'][i]['exposures'][ie]['dec'] = eoff.dec.deg
            
            # cosd = np.cos(plan['configs'][i]['exposures'][ie]['dec']/180*np.pi)
            # plan['configs'][i]['exposures'][ie]['ra'] += disp_offset/3600./cosd
            # plan['configs'][i]['exposures'][ie]['dec'] += spat_offset/3600.

            if prism_sel == 1:
                plan['configs'][i]['exposures'][ie]['gratingFilter'] = 'PRISM_CLEAR'

        plan['configs'][i]['exposures'] += prism_exposures

        plan['configs'][i]['primaryIds'] = sorted_ids
        plan['configs'][i]['fillerIds'] = []

        if keep_exposures is not None:
            NEXP = len(plan['configs'][i]['exposures'])
            for ie in range(NEXP)[::-1]:
                if ie not in keep_exposures:
                    plan['configs'][i]['exposures'].pop(ie)
    
    plan['stats'][0]['numberOfTargets'] = int(len(sorted_ids))

    plan['plannerSpecification']['gratingSpecification']['allowContamination'] = True

    if prism_sel == 2:
        
        plan['plannerSpecification']['gratingSpecification']['gratings'] = ["G395M_F290LP", "PRISM_CLEAR"]
        plan['plannerSpecification']['gratingSpecification']['allowContamination'] = True
        plan['plannerSpecification']['gratingSpecification']['multiplexLimit'] = None
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum'] = {}
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['empty'] = False
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['present'] = True
        
    elif prism_sel == 1:
        
        plan['plannerSpecification']['gratingSpecification']['gratings'] = ["PRISM_CLEAR"]
        plan['plannerSpecification']['gratingSpecification']['multiplexLimit'] = None
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum'] = {}
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['empty'] = True
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['present'] = False

    else:
        plan['plannerSpecification']['gratingSpecification']['gratings'] = ["G395M_F290LP"]

        plan['plannerSpecification']['gratingSpecification']['multiplexLimit'] = 4
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum'] = {}
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['empty'] = False
        plan['plannerSpecification']['gratingSpecification']['multiplexingMinimum']['present'] = True

    plan['plannerSpecification']['pointingSpecification']['partiallyCompletedSources'] = True

    plan['plannerSpecification']['slitSpecification']['sweetSpot'] = 'PERCENT_0'
    
    json_file = f'{name}.json'
    print(json_file)

    with open(json_file,'w') as fp:
        json.dump(plan, fp)

    if prism_sel == 2:
        pass

    elif prism_sel:
        # ! perl -pi -e "s/G395M\/F290LP/PRISM\/CLEAR/g" {name}.json
        # ! perl -pi -e "s/G395M_F290LP/PRISM_CLEAR/g" {name}.json
        # ! perl -pi -e "s/G395M/PRISM/g" {name}.json
        # ! perl -pi -e "s/g395m/prism/g" {name}.json
        
        repl = [
            ["G395M/F290LP", "PRISM/CLEAR"],
            ["G395M_F290LP", "PRISM_CLEAR"],
            ["G395M", "PRISM"],
            ["g395m", "prism"]
        ]
        _ = plan_utils.replace_strings_in_file(json_file, repl)
        
        ####! perl -pi -e "s/{name}/{name}_prism/g" {name}.json
    else:
        # ! perl -pi -e "s/PRISM_CLEAR/G395M_F290LP/g" {name}.json
        # ! perl -pi -e "s/JWST_NIRSPEC_PRISM/JWST_NIRSPEC_G395M/g" {name}.json

        repl = [
            ["PRISM_CLEAR", "G395M_F290LP"],
            ["JWST_NIRSPEC_PRISM", "JWST_NIRSPEC_G395M"],
        ]
        
        if grating.upper() != "G395M":
            repl += [
                ["G395M_F290LP", f"{grating}_{filter}".upper()],
                ["JWST_NIRSPEC_G395M", f"JWST_NIRSPEC_{grating}.upper()"]
            ]

        _ = plan_utils.replace_strings_in_file(json_file, repl)

    make_apt_shutter_csv(shutter_table, name=name)


class ShutterTable():
    def __init__(self, shutter_table, cat, nwave=64, smask=None, pointing_name="my_mask", pr=None, weight_column='Weight'):
        """
        """
        self.shutter_table = shutter_table
        self.cat = cat
        self.pointing_name = pointing_name
        
        self.weight_column = weight_column
        
        if smask is None:
            self.smask = shutters.ShutterMask()
        else:
            self.smask = smask
        
        if pr is None:
            self.pr = trace.PrismTrace()
        else:
            self.pr = pr

        self.set_waves(nwave=nwave)

        self.initialize_masks()
        
    def initialize_masks(self):
        
        self.olap_computed = False
        self.olap = np.zeros((self.N, self.N), dtype=bool)
        
        self.ok_prism = np.zeros(self.N, dtype=bool)
        self.ok_open = np.zeros(self.N, dtype=bool)
        self.ok_grating = np.zeros(self.N, dtype=bool)
        
        self.paths = []
        self.stuck_open_paths = []
        self.path_frac = np.ones(self.N)
        
        self.shutter_table['priority'] = self.priority
        self.shutter_table['ok_open'] = self.ok_open
        self.shutter_table['ok_prism'] = self.ok_prism
        self.shutter_table['ok_grating'] = self.ok_grating
    
    def set_waves(self, wlim=[0.6, 5.45], nwave=256):
        """
        """
        self.waves = np.linspace(*wlim, nwave)

    
    @property
    def meta(self):
        return self.shutter_table.meta

    
    @property
    def ix(self):
        return self.shutter_table['ix']


    @property
    def priority(self):
        return self.cat['Priority'][self.ix]

    
    @property
    def N(self):
        return len(self.shutter_table)

    
    def init_paths(self, verbose=True, pad=1):
        """
        """
        from tqdm import tqdm
        if verbose:
            print('Initialize slitlet paths')

        coarse_waves = np.linspace(*[0.7, 5.2], 80)
        fine_waves = np.linspace(*[0.6, 5.4], 256)

        import msaexp.utils
        coarse_waves = msaexp.utils.get_standard_wavelength_grid('prism', sample=0.03)
        fine_waves = coarse_waves
        
        self.paths = [self.pr.slitlet_path(waves=coarse_waves, pad=pad, **row)
                      for row in self.shutter_table]

        high_priority = self.priority < 2.5
        for j in np.where(high_priority)[0]:
            self.paths[j] = self.pr.slitlet_path(waves=fine_waves, pad=pad, **self.shutter_table[j])
            
        for j, p in enumerate(self.paths):
            fi = []
            for i in [0,1]:
                if p[i] is not None:
                    fi.append(p[i].frac)
            if len(fi) > 0:
                self.path_frac[j] = np.max(fi)

    
    def get_stuck_open(self, stuck_open=None, verbose=True):
        """
        Overlaps with stuck open shutters
        """
        
        if stuck_open is None:
            stuck_open = self.smask.stuck_open

        for s in stuck_open:
            q, r, c = np.cast[int](s.split())
            self.stuck_open_paths.append(self.pr.slitlet_path(waves=self.waves, shutter_quadrant=q,
                                    shutter_row=r, shutter_column=c, pad=0))

        if len(self.paths) == 0:
            self.init_paths()
        
        for i, pi in enumerate(self.paths):
            for op in self.stuck_open_paths:
                for k in [0,1]:
                    if (pi[k] is not None) & (op[k] is not None):
                        if pi[k].intersects_path(op[k]):
                            self.ok_open[i] = False
        
                if ~self.ok_open[i]:
                    break

        if verbose:
            print(f'Found {(~self.ok_open).sum()} slitlets that overlap with stuck-open')

    
    def compute_all_overlaps(self, verbose=True):
        """
        """
        from tqdm import tqdm
        
        if len(self.paths) == 0:
            self.init_paths()

        if verbose:
            print('Compute all overlaps')

        if self.N > 400:
            _iter = tqdm(range(self.N))
        else:
            _iter = range(self.N)
            
        for i in _iter:
            pi = self.paths[i]
            for j in range(i+1, self.N):
                pj = self.paths[j]
                for k in [0,1]:
                    if (pi[k] is not None) & (pj[k] is not None):
                        self.olap[i,j] |= pi[k].intersects_path(pj[k])
                        self.olap[j,i] |= self.olap[i,j]

        self.olap_computed = True


    def get_overlap_mask(self, mask_stuck_open=True, check_partial=True, verbose=True):
        """
        """
        
        ol = np.ones((self.N, self.N), dtype=bool)
    
        if check_partial:
            # Prefer sources within a particular priority that have most complete spectra
            un = utils.Unique(self.priority, verbose=False)
            for p in un.values:
                print('partial: ', p)
                pri = un[p] & self.ok_prism
                dup = (self.olap @ (np.eye(self.N)*pri)) > 0
                so = np.where(dup.sum(axis=0) > 0)[0]
                
                for j in so:
                    if (self.path_frac[j] < 1.) & (dup[j,:].sum() > 0):
                        if self.path_frac[j] < self.path_frac[dup[j,:]].max():
                            #print(f'P{p:.0f} remove {j}')
                            self.ok_prism[j] = False
                    
        ol[~self.ok_prism,:] = False
        ol[:,~self.ok_prism] = False
        ol &= self.olap
    
        # plt.imshow(ol*1)
        
        if mask_stuck_open:
            self.ok_prism &= self.ok_open | (self.priority <= 2.1)
            ol[~self.ok_open, :] = False
            ol[:, ~self.ok_open] = False
    
        if not self.olap_computed:
            self.compute_all_overlaps()
            
        # print(WEIGHT_COLUMN)
        wh = self.cat[self.weight_column][self.ix]
        so = np.argsort(wh)[::-1]
        
        for j in so:
            if wh[j] < 0:
                break
                
            #nj = (nrs1 & tru).sum(axis=0)
            oj = (ol)[j,:]
            #    oj &= self.ok_open
                
            self.ok_prism &= ~oj
            #ok_olap[~self.ok_prism] = False
            
            ol[~self.ok_prism,:] = False
            ol[:,~self.ok_prism] = False
    
        if verbose:
            print(f'Found {(~self.ok_prism).sum()} slitlets that overlap with a higher weight')


    def pad_grating(self, row_pad=[5,30], column_pad=3, valid=[1,2,3,4,5,6,7,8,9,10,11,12], mag_lim=30, verbose=True):
        """
        """
        
        self.ok_grating = (np.isin(self.priority, valid) & (self.cat['Magnitude'][self.ix] < mag_lim)) | self.ok_prism
        self.ok_grating &= self.priority != 11.
        
        #self.ok_grating = np.ones(self.N, dtype=bool)
        
        wh = self.cat[self.weight_column][self.ix]*1
        wh[self.ok_prism] += 100
        
        so = np.argsort(wh)[::-1]
        
        sh = self.shutter_table
        sh.meta['row_pad0'] = row_pad[0]
        sh.meta['row_pad1'] = row_pad[1]
        sh.meta['col_pad'] = column_pad

        stuck_open_mask = np.zeros(self.N, dtype=bool)
        for op in self.smask.stuck_open_table:
            match = (sh['shutter_quadrant'] == op['Q'])
            match &= (np.abs(sh['shutter_row'] - op['x']) < 8)
            match &= (np.abs(sh['shutter_column'] - op['y']) < 8)
            stuck_open_mask |= match

        if verbose:
            print(f'Remove {stuck_open_mask.sum()} slits close to stuck_open shutters')
        
        self.ok_grating &= ~stuck_open_mask
            
        rpad = np.ones(self.N)*row_pad[1]
        rpad[self.priority < 4] = row_pad[0]
        
        for j in so:
            if wh[j] < 0:
                break
            
            if not self.ok_grating[j]:
                continue
            
            row = sh['shutter_row'][j]
            column = sh['shutter_column'][j]
            quad = sh['shutter_quadrant'][j]

            match = (sh['shutter_quadrant'] == quad)
            match &= (np.abs(sh['shutter_row'] - row) < rpad)
            match &= (np.abs(sh['shutter_column'] - column) < column_pad)
            match &= self.ok_grating
            match[j] = False
            # print(j, wh[j], match.sum(), self.ok_grating.sum())
            
            self.ok_grating &= ~match

        if verbose:
            print(f'Trimmed {(~self.ok_grating).sum()} sources within grating padding of {row_pad} row and {column_pad} col shutters')

    
    def random_sky(self, NR=4096*2):
        """
        """
        sh = self.shutter_table

        used_slits = BAD_SHUTTERS_IX*1
        for roff in [3,2,1,0,-1]:
            used_slits = np.append(used_slits, np.cast[int](sh['shutter_quadrant']*1.e6 + 1000*sh['shutter_row'] + sh['shutter_column']+roff)[self.ok_prism])
            
        #NR = 4096*2
        qr = np.random.randint(1, high=4, size=NR, dtype=int)
        rr = np.random.randint(1, high=365, size=NR, dtype=int)
        cr = np.random.randint(1, high=171, size=NR, dtype=int)
        
        slits = np.cast[int](1e6*qr + 1000*rr + cr).tolist()
        bad_slits = np.in1d(slits, used_slits)
        for roff in [2,1,0,]:
            for coff in [0]: #[1,0,-1]:
                #xslits = [f'{qi} {c} {r}' for qi, r, c in zip(shut['quad'], rowi+roff, coli+coff)]
                xslits = np.cast[int](1e6*qr + 1000*(rr+coff) + cr+roff).tolist()
                bad_slits |= np.in1d(xslits, used_slits)
        
        xr = (rr[:,None] - sh['shutter_row'][self.ok_prism])
        xc = (cr[:,None] - sh['shutter_column'][self.ok_prism])
        xq = (qr[:,None] - sh['shutter_quadrant'][self.ok_prism])
        
        # bad_slits
        test = (xq == 0) & (np.abs(xr) < 150) & (np.abs(xc) < 3)
        bad_slits |= test.sum(axis=1) > 0
        
        sky = utils.GTable()
        sky['shutter_quadrant'] = qr
        sky['shutter_row'] = rr
        sky['shutter_column'] = cr
        sky['raw_row'] = rr
        sky['raw_col'] = rr
        sky['ix'] = 0
        
        sky = sky[~bad_slits]

        self.cat['Priority'][0] = 11
        self.cat[self.weight_column][0] = 1.0
        
        import msaexp.utils
        import astropy.table
        
        coarse_waves = msaexp.utils.get_standard_wavelength_grid('prism', sample=0.03)

        self.shutter_table = astropy.table.vstack([self.shutter_table, sky])
        
        print(f'Add {len(sky)} sky slits and reinitialize')
        
        self.initialize_masks()
    
        
    def plot_slitlets(self, figsize=(7,14), grating=False, att=None, draw_polygons=True):
        """
        make a plot
        """
        import pysiaf

        ok_test = self.ok_grating if grating else self.ok_prism
        
        fig, axes = plt.subplots(1,2,figsize=figsize, sharex=False, sharey=False)
        
        if draw_polygons:
            for p in self.stuck_open_paths:
                for i in [0,1]:
                    if p[i] is not None:
                        axes[i].add_patch(
                            PathPatch(p[i], fc='purple', alpha=0.5, ec='None', zorder=100)
                        )
        
        pri = np.ceil(self.priority)
        pri = np.floor(self.priority)
        
        for j, p in enumerate(self.paths):
            for i in [0,1]:
                if p[i] is not None:
                
                    if pri[j] == 1:
                        fc = PRIORITY_COLOR[1]
                        zo = 100
                    elif pri[j] == 0:
                        fc = PRIORITY_COLOR[0]
                        zo = 200
                    elif pri[j] == 2:
                        fc = PRIORITY_COLOR[2]
                        zo = 90
                    elif pri[j] == 3:
                        fc = PRIORITY_COLOR[3]
                        zo = 80
                    elif pri[j] == 11:
                        fc = PRIORITY_COLOR[11]
                        zo = 70
                    else:
                        fc = PRIORITY_COLOR['other']
                        zo = 10
        
                    ec = 'None'
                    alph = 0.5
                    
                    if ~ok_test[j]:
                        # alph = 0.1
                        ec = fc
                        fc = 'None'
                        zo -= 2
                        
                        if pri[j] < 4:
                            alph = 0.8
                        elif pri[j] > 5:
                            continue
                            
                    if self.ok_open[j]:
                        hatch = None
                    else:
                        hatch = '/////'
                        
                    if draw_polygons:
                        axes[i].add_patch(PathPatch(p[i], fc=fc, ec=ec, hatch=hatch, alpha=alph, zorder=zo))
        
        for ax in axes:
        
            ax.set_ylim(-40, 2049)
            ax.set_yticks(np.arange(0,2049,256, dtype=int))
            ax.set_xticks(np.arange(0,2049,256, dtype=int))
            
            ax.grid()
        
        if att is not None:
            siaf = pysiaf.Siaf('Nirspec')
            dets = [siaf[f'NRS{d}_FULL'] for d in [1,2]]

            inq = self.cat['quad'] > 0
            un = utils.Unique(self.cat[inq]['Priority'], verbose=False)

            for i, d in enumerate(dets):
                d.set_attitude_matrix(att)
                xi, yi = d.sky_to_sci(self.cat[inq]['RA'], self.cat[inq]['Dec'])
                axes[i].scatter(xi[un[0.]], yi[un[0.]], color='magenta', marker='.', alpha=0.5)
                axes[i].scatter(xi[un[1.]], yi[un[1.]], color='r', marker='.', alpha=0.5)
                axes[i].scatter(xi[un[2.]], yi[un[2.]], color='orange', marker='.', alpha=0.5)
                axes[i].scatter(xi[un[3.]], yi[un[3.]], color='steelblue', marker='.', alpha=0.5)
                # axes[i].scatter(xi[un[4.]], yi[un[4.]], color='purple', marker='.', alpha=0.5)

        axes[1].text(0.95, 0.95, self.pointing_name,
                     ha='right', va='top', transform=axes[1].transAxes)
        
        axes[0].set_xlim(500, 2049)
        axes[1].set_xlim(0, 2049-500)
        axes[1].set_yticklabels([])

        # Legend
        ax = axes[0]

        for i, k in enumerate(PRIORITY_COLOR):
            ax.plot([800,800], np.ones(2)*900+20*i, color=PRIORITY_COLOR[k], alpha=0.5, lw=8, 
                    label=f'Priority: {k}')
        
        ax.legend(loc='center left')

        fig.tight_layout(pad=2)

        return fig

    
    def summary(self):
        """
        Print a summary of valid stlits
        """
        un = utils.Unique(self.cat['Priority'], verbose=False)

        self.cat['has_shutter'] = False
        self.cat['has_shutter'][self.ix] = True
        
        ix_prism = self.ix[self.ok_prism]
        ix_prism_src = self.ix[self.ok_prism & (self.priority != 11)]
        ix_grating = self.ix[self.ok_grating]
        ix_both = self.ix[self.ok_prism & self.ok_grating]

        lines = []
        lines.append("# p   Np  pure  grat  both")
        test = self.cat['has_shutter']
        lines.append(f"# N  {test.sum():>4} {test[ix_prism_src].sum():>4} {test[ix_grating].sum():>4}  {test[ix_both].sum():>4}")
        lines.append(f'# {self.pointing_name}')
        
        for v in un.values:
            test = un[v] & self.cat['has_shutter']
            lines.append(f"P{v:<3.0f} {test.sum():>4} {test[ix_prism].sum():>4} {test[ix_grating].sum():>4}  {test[ix_both].sum():>4}")

        lines[-1] += '\n'
        
        print('\n'.join(lines))
        with open(f'{self.pointing_name}.summary.txt','w') as fp:
            fp.write('\n'.join(lines))
    
    def write_outputs(self, att=None, make_figures=False, grating='g395m', filter='f290lp', **kwargs):

        sky = self.priority == 11
        if sky.sum() > 0:
            print('Set sky IDs: ', sky.sum())
            self.shutter_table['id'][sky] = (500000 + np.arange(sky.sum())).astype(int)

        self.shutter_table['priority'] = self.priority
        self.shutter_table['ok_open'] = self.ok_open
        self.shutter_table['ok_prism'] = self.ok_prism
        self.shutter_table['ok_grating'] = self.ok_grating
            
        self.shutter_table.write(f'{self.pointing_name}_shutter_table.fits', overwrite=True)
        
        if make_figures:
            fig = self.plot_slitlets(att=att)
            fig.savefig(f'{self.pointing_name}_prism.png')
            
            fig = self.plot_slitlets(att=att, grating=True)
            fig.savefig(f'{self.pointing_name}_{grating}.png')

        _ = make_plan_json(
            self.shutter_table[self.ok_grating & (self.priority != 11)], 
            name=f'{self.pointing_name}_{grating}',
            prism_sel=(grating.lower() == "prism"),
            grating=grating,
            filter=filter,
            **kwargs
        )

        # _ = make_plan_json(self.shutter_table, name=f'{self.pointing_name}_full', **kwargs)
        
        # if grating == "prism":
        #     _ = make_plan_json(
        #         self.shutter_table[self.ok_prism & (self.priority != 11)],
        #         name=f'{self.pointing_name}_prism',
        #         **kwargs
        #     )
        # else:
        #     _ = make_plan_json(
        #         self.shutter_table[self.ok_grating & (self.priority != 11)],
        #         name=f'{self.pointing_name}_{grating}',
        #         prism_sel=False,
        #         grating=grating,
        #         filter=filter,
        #         **kwargs
        #     )

        # _ = make_plan_json(self.shutter_table[self.ok_prism & (self.priority < 3)], name=f'{self.pointing_name}_prism_p12', **kwargs)
        
        # if 11 in self.priority:
        #     _ = make_plan_json(self.shutter_table[self.ok_prism], name=f'{self.pointing_name}_prism_bkg', **kwargs)
            
        # _ = make_plan_json(self.shutter_table[self.ok_prism], name=f'{self.pointing_name}_both', prism_sel=2)
        

        
        
