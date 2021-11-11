import math

from gigalens.tf.profiles.light import sersic


def test_sersic_ellipse():
    se = sersic.SersicEllipse(use_lstsq=False)
    half_light = 5.0
    a = se.light(
        x=0.0,
        y=1.0,
        R_sersic=1.0,
        n_sersic=2.0,
        center_x=0.0,
        center_y=0.0,
        e1=0.0,
        e2=0.0,
        Ie=5.0,
    )
    assert math.isclose(a, half_light), "Half light amplitude"
