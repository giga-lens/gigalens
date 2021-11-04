from abc import ABC

from gigalens.profiles.parameterized import Parameterized


class MassProfile(Parameterized, ABC):
    def deriv(self, x, y, params):
        return NotImplemented
