import numpy as np
import os
from pathlib import Path
from tempfile import gettempdir
import hashlib
from examples.seismic import SeismicModel, RickerSource, Receiver
from devito import *

v_p_water, v_p_0, v_p_1 = 1.48, 1.515, 1.650
v_p_v_s_ratio = 4.
density_water, density_0, density_1 = 1.0, 1.9, 2.1

class Model:
    def __init__(self, n_x, n_z, x_0=0., z_0=0.):
        self.so, self.to = 4, 2

        self.x = x_0 + np.arange(n_x) * self.d_x
        self.z = z_0 + np.arange(n_z) * self.d_z
        self.n_x = n_x
        self.n_z = n_z
        self.x_0 = x_0
        self.z_0 = z_0

        self.t_0 = 0.

        self.result = None

    def naive_hash(self):
        vars = self.so, self.to, self.x.tolist(), self.z.tolist(), self.n_x, self.n_z, self.x_0, self.z_0, self.a_x, self.a_z, self.seed
        return get_hash(vars)

    def __repr__(self):
        return self.naive_hash()

    def devito_model(self, demo=False, nbl=50, x_0=None, z_0=None, mode='elastic'):
        if x_0 is None:
            x_0 = self.x_0
        if z_0 is None:
            z_0 = self.z_0

        v_p, v_s, b = self.elastic_model()
        if mode == 'acoustic':
            v_s = np.full(v_s.shape, 0.)
        model = SeismicModel(
            vp=v_p,
            vs=v_s,
            b=b,
            space_order=self.so,
            origin=(x_0, z_0),
            shape=v_p.shape,
            spacing=(self.d_x, self.d_z),
            nbl=nbl, bcs='mask'
        )

        model.dt_scale = 0.9

        # Fix bug in initialising damping field
        try:
            model.update('damp', fix_damping_field(model.damp.data))
            assert model.damp.data.min() >= 0
            assert model.damp.data.max() <= 1
        except AttributeError:
            # Damping field is just 0 or 1
            pass

        model.parent = self

        return model

    def realize_model(self):
        raise NotImplementedError

    def elastic_model(self):
        raise NotImplementedError

    def acoustic_reflectivity_model(self):
        """Returns the acoustic reflectivity based on the P-wave velocity and density."""

        v_p, _, density = self.elastic_model()
        Z = v_p * density # Acoustic impedance

        return (Z[:, 1:] - Z[:, :-1]) / (Z[:, 1:] + Z[:, :-1])


def fix_damping_field(a):
    return (a - a.min()) / (a.max() - a.min())

def get_hash(*args):
    args = list(args)
    return hashlib.md5(' '.join([str(x) for x in args]).encode()).hexdigest()

def get_cache_path(name, prefix='sbp', extension='npz'):
    try:
        tmp = Path(os.environ['SBP_CACHE_PATH'])
    except KeyError:
        print('SBP_CACHE_PATH unset')
        tmp = Path(gettempdir())
    p = tmp / (prefix + '_' + str(name) + '.' + extension)
    return p

def run_single_channel_modelling(model, source_f_0, source_xz, time_range, d_t_resample=1e-2, time_order=4, weak_caching=False, debug=False):
    assert isinstance(model, SeismicModel)

    h = get_hash(
           (model.lam.data * model.mu.data * model.b.data).tolist(),
           source_f_0, source_xz,
           time_range.time_values,
           d_t_resample,
           time_order)
    cache_path = get_cache_path(h)

    try:
        load = np.load(cache_path)
        result = load['t'], load['trace']
        print(f'Loaded Devito run from {cache_path}')
    except FileNotFoundError:
        print(f'Failed to load Devito run from {cache_path}, re-running')

        time = model.grid.time_dim
        s = time.spacing

        v = VectorTimeFunction(name='v', grid=model.grid,
                               space_order=model.space_order, time_order=time_order)
        tau = TensorTimeFunction(name='t', grid=model.grid,
                                 space_order=model.space_order, time_order=time_order)

        src = RickerSource(name='src', grid=model.grid, f0=source_f_0, time_range=time_range)
        src.coordinates.data[:] = source_xz

        src_xx = src.inject(field=tau.forward[0, 0], expr=s * src)
        src_zz = src.inject(field=tau.forward[1, 1], expr=s * src)

        if debug:
            rec = Receiver(name='rec', grid=model.grid, npoint=model.shape[0], time_range=time_range)
        else:
            rec = Receiver(name='rec', grid=model.grid, npoint=1, time_range=time_range)

        # Offset the receiver by a grid point to avoid weird numerical artefacts with co-located source
        if debug:
            rec.coordinates.data[:, 0] = model.grid.spacing[0] * np.arange(model.shape[0])
        else:
            rec.coordinates.data[:, 0] = src.coordinates.data[:, 0] + 2 * model.grid.spacing[0]
        rec.coordinates.data[:, 1] = src.coordinates.data[:, 1]

        rec_term = rec.interpolate(expr=tau[0, 0] + tau[1, 1])  # Pressure?

        # Lame parameters
        l, mu, ro = model.lam, model.mu, model.b

        # fdelmodc reference implementation
        u_v = Eq(v.forward, model.damp * (v + s * ro * div(tau)))
        u_t = Eq(tau.forward, model.damp * (tau + s * (l * diag(div(v.forward)) +
                                                       mu * (grad(v.forward) + grad(v.forward).T))))

        # opt=('advanced', {'openmp': True})
        op = Operator([u_v] + [u_t] + src_xx + src_zz + rec_term)
        op(dt=time_range.step)
        resample = rec.resample(dt=d_t_resample)
        if debug:
            result = resample.time_values[:-1], np.array(resample.data)
        else:
            result = resample.time_values[:-1], np.array(resample.data).ravel()[:-1]
        np.savez_compressed(cache_path, t=result[0], trace=result[1])
    finally:
        return result
