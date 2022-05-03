import math
import numpy as np
from lenstronomy.LensModel.Profiles import epl as LenstronomyEPL, shear as LenstronomyShear, sie as LenstronomySIE, \
    sis as LenstronomySIS
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet as LenstronomyShapelets

import gigalens.jax.profiles.light
import gigalens.jax.profiles.mass
import gigalens.tf.profiles.light
import gigalens.tf.profiles.mass


def test_sersic_ellipse():
    for se in [gigalens.tf.profiles.light.sersic.SersicEllipse(use_lstsq=False),
               gigalens.jax.profiles.light.sersic.SersicEllipse(use_lstsq=False)]:
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
    for shp in [gigalens.tf.profiles.light.shapelets.Shapelets(n_max=5, use_lstsq=False, interpolate=True),
                gigalens.tf.profiles.light.shapelets.Shapelets(n_max=5, use_lstsq=False, interpolate=False),
                gigalens.jax.profiles.light.shapelets.Shapelets(n_max=5, use_lstsq=False, interpolate=True),
                gigalens.jax.profiles.light.shapelets.Shapelets(n_max=5, use_lstsq=False, interpolate=False), ]:
        amplitudes = np.random.normal(size=(shp.n_layers, 1)).astype(np.float32)
        amp_dict = {name: x for name, x in zip(shp._amp_names, amplitudes)}

        x, y = np.random.normal(size=(5, 5, 1)).astype(np.float32), np.random.normal(size=(5, 5, 1)).astype(np.float32)
        a = np.asarray(shp.light(x=x, y=y, center_x=0, center_y=0, beta=1, **amp_dict))
        b = LenstronomyShapelets().function(x=x.flatten(), y=y.flatten(), center_x=0, center_y=0, beta=1,
                                            amp=amplitudes.flatten(), n_max=shp.n_max)
        assert np.allclose(a.flatten(), b.flatten(), rtol=1e-5, atol=1e-4), "Lenstronomy parity"


def test_epl():
    for gl_model in [gigalens.tf.profiles.mass.epl.EPL(100),
                     gigalens.jax.profiles.mass.epl.EPL(100)]:
        le_model = LenstronomyEPL.EPL()
        x, y = np.random.normal(size=10000).astype(np.float32), np.random.normal(size=10000).astype(np.float32)
        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1.0, gamma=2.0, e1=0., e2=0., center_x=0., center_y=0.)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1, gamma=2, e1=0, e2=0, center_x=0, center_y=0)
        assert np.allclose(gl_deriv[0], le_deriv[0], rtol=1e-5, atol=1e-4)
        assert np.allclose(gl_deriv[1], le_deriv[1], rtol=1e-5, atol=1e-4)

        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1.2, gamma=2.2, e1=-0.1, e2=0.1, center_x=0., center_y=0.)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1.2, gamma=2.2, e1=-0.1, e2=0.1, center_x=0, center_y=0)
        assert np.allclose(gl_deriv[0], le_deriv[0], rtol=1e-5, atol=1e-4)
        assert np.allclose(gl_deriv[1], le_deriv[1], rtol=1e-5, atol=1e-4)


def test_sis():
    for gl_model in [gigalens.tf.profiles.mass.sis.SIS(),
                     gigalens.jax.profiles.mass.sis.SIS()]:
        le_model = LenstronomySIS.SIS()
        x, y = np.random.normal(size=10000).astype(np.float32), np.random.normal(size=10000).astype(np.float32)
        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1., center_x=0., center_y=0.)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1., center_x=0., center_y=0.)
        assert np.allclose(gl_deriv[0], le_deriv[0])
        assert np.allclose(gl_deriv[1], le_deriv[1])

        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1.2, center_x=0., center_y=0.)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1.2, center_x=0, center_y=0)
        assert np.allclose(gl_deriv[0], le_deriv[0])
        assert np.allclose(gl_deriv[1], le_deriv[1])


def test_sie():
    for gl_model in [gigalens.tf.profiles.mass.sie.SIE(),
                     gigalens.jax.profiles.mass.sie.SIE()]:
        le_model = LenstronomySIE.SIE()
        x, y = np.random.normal(size=10000).astype(np.float32), np.random.normal(size=10000).astype(np.float32)
        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1., center_x=0., center_y=0., e1=1e-3, e2=1e-3)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1, center_x=0, center_y=0, e1=1e-3, e2=1e-3)
        assert np.allclose(gl_deriv[0], le_deriv[0], rtol=1e-5, atol=1e-4)
        assert np.allclose(gl_deriv[1], le_deriv[1], rtol=1e-5, atol=1e-4)

        gl_deriv = gl_model.deriv(x=x, y=y, theta_E=1.2, center_x=0., center_y=0., e1=0.1, e2=-0.1)
        le_deriv = le_model.derivatives(x=x, y=y, theta_E=1.2, center_x=0, center_y=0, e1=0.1, e2=-0.1)
        assert np.allclose(gl_deriv[0], le_deriv[0], rtol=1e-5, atol=1e-4)
        assert np.allclose(gl_deriv[1], le_deriv[1], rtol=1e-5, atol=1e-4)


def test_shear():
    for gl_model in [gigalens.tf.profiles.mass.shear.Shear(),
                     gigalens.jax.profiles.mass.shear.Shear()]:
        le_model = LenstronomyShear.Shear()
        x, y = np.random.normal(size=10000).astype(np.float32), np.random.normal(size=10000).astype(np.float32)
        gl_deriv = gl_model.deriv(x=x, y=y, gamma1=0., gamma2=0.)
        le_deriv = le_model.derivatives(x=x, y=y, gamma1=0., gamma2=0.)
        assert np.allclose(gl_deriv[0], le_deriv[0])
        assert np.allclose(gl_deriv[1], le_deriv[1])

        gl_deriv = gl_model.deriv(x=x, y=y, gamma1=0.1, gamma2=0.1)
        le_deriv = le_model.derivatives(x=x, y=y, gamma1=0.1, gamma2=0.1)
        assert np.allclose(gl_deriv[0], le_deriv[0])
        assert np.allclose(gl_deriv[1], le_deriv[1])
