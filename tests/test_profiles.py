import math
import numpy as np
from gigalens.tf.profiles.light import sersic, shapelets
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet as LenstronomyShapelets


def test_sersic_ellipse():
    se = sersic.SersicEllipse(use_lstsq=False)
    half_light = 5.0
    light_params = dict(R_sersic=1.0,
                        n_sersic=2.0,
                        center_x=0.0,
                        center_y=0.0,
                        e1=0.0,
                        e2=0.0,
                        Ie=half_light)
    a = se.light(x=0.0, y=1.0, **light_params)
    assert math.isclose(a, half_light), "Half light amplitude"

    x, y = np.random.normal(size=1000).astype(np.float32), np.random.normal(size=1000).astype(np.float32)
    a = se.light(x=x, y=y, **light_params)
    light_params.pop('Ie')
    b = SersicElliptic().function(x=x, y=y, **{'amp': half_light, **light_params})
    assert np.allclose(a, b), "Lenstronomy parity"


def test_shapelets():
    shp = shapelets.Shapelets(n_max=5, use_lstsq=False)
    amplitudes = np.random.normal(size=(1, shp.n_layers)).astype(np.float32)
    amp_dict = {name: x for name, x in zip(shp._amp_names, amplitudes)}

    x, y = np.random.normal(size=(5, 5, 1)).astype(np.float32), np.random.normal(size=(5, 5, 1)).astype(np.float32)
    a = shp.light(x=x, y=y, center_x=0, center_y=0, beta=1, **amp_dict).numpy()
    b = LenstronomyShapelets().function(x=x.flatten(), y=y.flatten(), center_x=0, center_y=0, beta=1,
                                        amp=amplitudes.flatten(), n_max=shp.n_max)
    assert np.allclose(a.flatten(), b.flatten(), rtol=1e-5, atol=1e-4), "Lenstronomy parity"
