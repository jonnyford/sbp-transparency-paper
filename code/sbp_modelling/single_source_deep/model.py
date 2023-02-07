import gstools as gs
import numpy as np
from sbp_modelling.model import Model, get_cache_path, v_p_water, v_p_0, v_p_1, v_p_v_s_ratio, density_water, density_0, density_1

x_0 = z_0 = 0
n_x = 3201
n_z = 3201

d_x = d_z = 0.025

class SingleSourceModel(Model):
    def __init__(self, a_x=1., a_z=1., seed=1):
        self.d_x = d_x
        self.d_z = d_z

        super().__init__(n_x, n_z)

        self.a_x = a_x
        self.a_z = a_z
        self.seed = seed

        self.h = self.n_z // 10

        self.binary_model = None

    def realize_binary_field(self, cache=True):
        if self.binary_model is None:
            h = self.naive_hash()
            cache_path = get_cache_path(h)
            try:
                if cache is not True:
                    raise FileNotFoundError
                self.binary_model = np.load(cache_path)['f']
                print(f'Loaded from cache {cache_path}')
            except FileNotFoundError:
                corr = gs.Exponential(dim=2, var=2, len_scale=[self.a_x, self.a_z])
                srf = gs.SRF(corr, seed=self.seed)
                srf.structured([self.x, self.z[:self.h]])
                gs.transform.binary(srf, upper=0, lower=1)

                self.binary_model = srf.field.astype(bool)
                #self.binary_model[:] = True # Debugging
                #self.binary_model[:, :self.n_top_layer] = self.binary_model[:, -self.n_top_layer:] = False
                np.savez_compressed(cache_path, f=self.binary_model)
        return self.binary_model

    def horizons(self, return_gridpoints=False):
        n = self.h
        waterbottom = n * 7
        mtd_top = n * 8 + 1
        mtd_base = n * 9 + 1

        if return_gridpoints is True:
            return waterbottom, mtd_top, mtd_base
        else:
            return waterbottom * self.d_z, mtd_top * self.d_z, mtd_base * self.d_z

    def realize_model(self, **kwargs):
        grid = np.zeros((self.n_x, self.n_z), dtype=int) # Lithology 1
        waterbottom, mtd_top, mtd_base = self.horizons(return_gridpoints=True)
        grid[:, :waterbottom] = -1 # Water
        grid[:, mtd_top:mtd_base] = self.realize_binary_field(**kwargs) # Random field
        return grid

    def elastic_model(self, **kwargs):
        grid = self.realize_model(**kwargs)
        # Water
        v_p = np.where(grid == -1, v_p_water, grid)
        b = np.where(grid == -1, density_water, grid)

        # Lithology 1
        v_p = np.where(grid == 0, v_p_0, v_p)
        b = np.where(grid == 0, density_0, b)

        # Lithology 2
        v_p = np.where(grid == 1, v_p_1, v_p)
        b = np.where(grid == 1, density_1, b)

        v_s = v_p / v_p_v_s_ratio
        v_s = np.where(grid == -1, 0., v_s)

        return v_p, v_s, b

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a_x, a_z, seed = 1, 0.05, 1001
    model = SingleSourceModel(a_x, a_z, seed)
    plt.figure()
    plt.pcolormesh(
        model.x, model.z, model.elastic_model(cache=False)[1].T,
        shading='nearest', cmap='viridis', rasterized=True
    )
    plt.colorbar()
    plt.show()

    print(model.devito_model().critical_dt)
    print(model.devito_model().critical_dt * model.devito_model().dt_scale)
