from dolfin import *
from dolfin_adjoint import *

class TidalParameters(object):
    # parameters (see Table 3.4 in https://link.springer.com/book/10.1007/978-3-319-59483-5)

    def __init__(self):

        self._bottom_friction = Constant(0.0025)
        self._viscosity = Constant(5.0)
        self._depth = Constant(50.0)
        self._thrust_coefficient = Constant(0.6)
        self._turbine_cross_section = Constant(314.15)

    @property
    def depth(self):
        "h"
        return self._depth

    @property
    def viscosity(self):
        "nu"
        return self._viscosity

    @viscosity.setter
    def viscosity(self, value):
        self._viscosity = value

    @property
    def thrust_coefficient(self):
        "C_t"
        return self._thrust_coefficient

    @property
    def turbine_cross_section(self):
        "A_t"
        return self._turbine_cross_section

    @property
    def bottom_friction(self):
        "c_b"
        return self._bottom_friction

    @bottom_friction.setter
    def bottom_friction(self, value):
        self._bottom_friction = value

    @property
    def gravity(self):
        "g"
        return Constant(9.81)

    @property
    def water_density(self):
        "rho"
        return 1025.0 # https://github.com/OpenTidalFarm/OpenTidalFarm/blob/ca1aa59ee17818dc3b1ab94a9cbc735527fb2961/opentidalfarm/problems/steady_sw.py#L60

    @property
    def initial_condition(self):
        """State initial conditions for SW solver
        source: https://zenodo.org/record/224251
        """
        return Constant(("1e-7", "0.0", "0.0"))

    @property
    def source_term(self):
        """PDE source term f_u"""
        return Constant((0.0, 0.0))


    def __str__(self):
        s = """
Water depth: {}
Viscosity: {}
Thrust coefficient: {}
Turbine cross section: {}
Natural bottom friction: {}
Gravity: {}
Water density: {}
            """.format(self.depth.values()[0], self.viscosity.values()[0],
                    self.thrust_coefficient.values()[0], self.turbine_cross_section.values()[0],
                    self.bottom_friction.values()[0], self.gravity.values()[0], self.water_density)
        return s



if __name__ == "__main__":
    parameters = TidalParameters()
    print(parameters)

