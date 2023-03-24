import os

import arviz as az
import pymc as pm
import pymc.sampling_jax


def softmax_regression(coords, save_model=None, cores=4):
    """

    :param y: one-hot encoded class for each observation, shape (n_observations, k classes)
    :param x: feature data for each observation, shape (n_observations, n_features)
    :param save_model:
    :param cores:
    :return:
    """

    if cores == 1:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

    x = coords.pop('x')
    y = coords.pop('y')

    with pm.Model(coords=coords) as sft_model:
        # order: [climate, from_crop_price, to_crop_price]
        a = pm.Normal('a', mu=0, sigma=0.5, dims='labels')
        coeff = pm.Normal('b', mu=0, sigma=0.5, dims=('features', 'labels'))

        theta = a + pm.math.dot(x, coeff)

        z = pm.math.softmax(theta)

        obs = pm.Categorical('obs', p=z, observed=y)

        trace = pm.sampling_jax.sample_numpyro_nuts()

        if save_model:
            az.to_netcdf(trace, save_model)
            print(save_model)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
