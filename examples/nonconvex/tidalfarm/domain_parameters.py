
class DomainParameters(object):
    # parameters (see p. 96 in https://link.springer.com/book/10.1007/978-3-319-59483-5)

    def __init__(self):
        pass

    @property
    def x_min(self):
        return 0.0

    @property
    def x_max(self):
        return 2000.0

    @property
    def y_min(self):
        return 0.0

    @property
    def y_max(self):
        return 1000.0

    @property
    def n(self):
        "space discretization parameter."
        return 100

