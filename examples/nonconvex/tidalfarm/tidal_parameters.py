
class TidalParameters(object):
    # parameters (see Table 3.4 in https://link.springer.com/book/10.1007/978-3-319-59483-5)

    def __init__(self):
        pass

    @property
    def depth(self):
        return 50.0

    @property
    def viscosity(self):
        return 5.0

    @property
    def thrust_coefficient(self):
        return 0.6

    @property
    def turbine_cross_section(self):
        return 314.15

    @property
    def bottom_friction(self):
        return 0.0025

    @property
    def gravity(self):
        return 9.81

    @property
    def water_density(self):
        return 1025.0 # https://github.com/OpenTidalFarm/OpenTidalFarm/blob/ca1aa59ee17818dc3b1ab94a9cbc735527fb2961/opentidalfarm/problems/steady_sw.py#L60
