import gstools as gs
import numpy as np
import os
import pandas as pd
from sbp_modelling.model import Model, get_cache_path, v_p_water, v_p_0, v_p_1, v_p_v_s_ratio, density_water, density_0, density_1
from scipy.ndimage import gaussian_filter

local_path = os.path.dirname(__file__)

data_path = os.path.join(local_path, '../../../data/multi-source')
horizon_path = os.path.join(data_path, 'all-horizons-crossline.dat')

df = pd.read_csv(horizon_path, names=('horizon', 'line', 'x', 'y', 'shot_id', 'z'), sep='\s+')
df['shot_id'] = (df['shot_id'] - 2109876480.00) / 1e10
df['shot_id'] = df['shot_id'].astype(int)
first_shot = df['shot_id'].idxmin()
df['offset'] = np.sqrt(np.sum(np.square((df[['x', 'y']] - df[['x', 'y']].loc[first_shot])), axis=1))

x_0, x_1 = 0, round(df['offset'].max(), -3)
z_0, z_1 = 800, 960
d_x = d_z = 0.05 #0.1 #0.15
n_x = int((x_1 - x_0) // d_x)
n_z = int((z_1 - z_0) // d_z)
x = x_0 + np.arange(n_x) * d_x
z = z_0 + np.arange(n_z) * d_z
pad = 50 # 40 # 50

t_0 = 0.

def get_horizon(df, hz, n=200):
    horizon = df[['offset', 'z']].loc[df['horizon'] == hz]
    horizon['z'] = horizon['z'].rolling(window=n, center=True, min_periods=1).mean()
    return horizon

def fourier_shift(arr, d_z, shift):
    """2-D shift to follow a surface."""

    k_z = np.fft.fftfreq(arr.shape[0], d=d_z)
    shift_k = np.exp(2j * np.pi * k_z[:, np.newaxis] * shift)

    return np.fft.ifft(np.fft.fft(arr, axis=0) * shift_k, axis=0).real

class MultiSourceModel(Model):
    def __init__(self, a_x=1., a_z=1., seed=1, left=None, right=None, top=None, base=None, subsampling=None):
        if subsampling is None:
            self.d_x = d_x
            self.d_z = d_z
        else:
            self.d_x, self.d_z = subsampling

        if left is None:
            left = x_0
        if right is None:
            right = x_1

        if top is None:
            top = z_0
        if base is None:
            base = z_1
        top = self.d_z * (top // self.d_z + 1)
        base = self.d_z * (base // self.d_z - 1)

        assert left >= x_0
        assert right <= x_1
        assert top >= z_0
        assert base <= z_1

        self.top = top

        n_x = int(1 + (right - left) // self.d_x)
        n_z = int(1 + (base - top) // self.d_z)

        super().__init__(n_x, n_z, x_0=left, z_0=top)

        self.a_x = a_x
        self.a_z = a_z
        self.seed = seed
        self.no_aniso = 1e7  # "Infinite" lateral correlation length (horizontal bedding)

        self.binary_model = None

        self.vertical_correction = self.z_0 - z_0

        #print(n_x, self.x.size, n_z, self.z.size, self.d_x, self.d_z, self.a_x, self.a_z, self.seed)

    def horizons(self):
        waterbottom = np.interp(self.x, *get_horizon(df, 'Seabed').to_numpy().T)

        mtd_base = get_horizon(df, 'MTD_base')
        mtd_x = (self.x >= mtd_base['offset'].min()) & (self.x <= mtd_base['offset'].max())
        mtd_base = np.interp(self.x, *mtd_base.to_numpy().T)

        mtd_top = np.interp(self.x, *get_horizon(df, 'MTD_top').to_numpy().T)
        mtd_z = (self.z >= mtd_top.min()) & (self.z <= mtd_base.max())

        return waterbottom, mtd_x, mtd_z, mtd_top, mtd_base

    def realize_model(self, cache=True):
        if self.binary_model is None:
            h = self.naive_hash()
            cache_path = get_cache_path(h, prefix='sbp-ternary')
            try:
                if cache is not True:
                    raise FileNotFoundError
                self.binary_model = np.load(cache_path)['f']
                print(f'Loaded from cache {cache_path}')
            except (FileNotFoundError, ValueError):
                waterbottom, mtd_x, mtd_z, mtd_top, mtd_base = self.horizons()

                sediment_corr = gs.Exponential(dim=2, var=2, len_scale=[self.a_z, self.no_aniso])
                sediment_srf = gs.SRF(sediment_corr, seed=self.seed)
                sediment_srf.structured([self.z - self.vertical_correction, [x_0, ]])
                gs.transform.binary(sediment_srf, upper=0, lower=1)

                self.binary_model = np.tile(sediment_srf.field.astype(int), (1, self.n_x))

                mtd_thickness = mtd_base - mtd_top

                mtd_corr = gs.Exponential(dim=2, var=2, len_scale=[self.a_z, self.a_x])
                mtd_srf = gs.SRF(mtd_corr, seed=self.seed)
                mtd_srf.structured([self.z[:mtd_z.sum()] - self.vertical_correction, self.x[mtd_x]]) # zs need to match sediment_srf
                gs.transform.binary(mtd_srf, upper=0, lower=1)

                # We don't necessarily have any MTD in every shot, check first
                if mtd_x.sum() > 0:
                    n = int(mtd_thickness[mtd_x].mean() // self.d_z)
                    self.binary_model[:n, mtd_x] = mtd_srf.field[:n]

                np.savez_compressed(cache_path, f=self.binary_model)
        return self.binary_model

    def water_layer(self, smooth=1.):
        waterbottom, mtd_x, mtd_z, mtd_top, mtd_base = self.horizons()
        water_model = np.tile(self.z[:, np.newaxis], (1, self.x.size)) - waterbottom
        water_model = 0.5 + np.clip(water_model, -smooth / 2, smooth / 2) / smooth
        water_model = 1 - gaussian_filter(water_model, sigma=smooth / (2 * self.d_x))

        return water_model

    def elastic_model(self, ind_x=None, **kwargs):
        self.realize_model(**kwargs)

        if ind_x is None:
            m = self.binary_model
        else:
            m = self.binary_model[:, ind_x]

        v_p = np.where(m == 1, v_p_1, v_p_0)  # sediments
        b = np.where(m == 1, density_1, density_0)

        waterbottom = self.horizons()[0]

        v_p = 1 / fourier_shift(1 / v_p, self.d_z, self.z[0] - waterbottom)
        b = fourier_shift(b, self.d_z, self.z[0] - waterbottom)
        v_s = v_p / v_p_v_s_ratio

        water_model = self.water_layer(smooth=self.d_z)

        v_p = np.divide(1, (water_model / v_p_water) + ((1 - water_model) / v_p))  # add water layer
        v_s = np.divide(1, (water_model / 0.1) + ((1 - water_model) / v_s))  # v_s = 100 ms-1 in water
        v_s = np.where(v_s == 0.1, 0.0, v_s)
        b = np.divide(1, (water_model / density_water) + ((1 - water_model) / b))

        return v_p.T, v_s.T, b.T

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = MultiSourceModel(1., 0.05, 1001, left=3000, right=3100, top=900, base=z_1) #, left=2000, right=3000)
    model.realize_model()
    fig, ax = plt.subplots(1, 1)
    #plot_velocity(model.devito_model(nbl=50))
    plt.imshow(model.devito_model(nbl=0).mu.data, aspect='auto', interpolation='bicubic')
    plt.colorbar()
    fig.show()
