__all__ = ["BoundaryConditionSet"]

# source: https://github.com/OpenTidalFarm/OpenTidalFarm/blob/master/opentidalfarm/boundary_conditions.py

class BoundaryConditionSet(list):
    """ Stores a list of boundary conditions.
    Expression with an attribute named t will be
    automatically updated to the current timestep during the simultion.
    """

    def update_time(self, t, only_type=None, exclude_type=None):
        ''' Update the time attribute for all boundary conditions '''

        if exclude_type is None:
            exclude_type = []

        for bc in self:
            if only_type is not None and bc[-1] not in only_type:
                continue
            if bc[-1] in exclude_type:
                continue

            if hasattr(bc[1], "t"):
                bc[1].t = t

    def add_bc(self, function_name, expression=None, facet_id=None, bctype="strong_dirichlet"):
        """ Valid choices for bctype: "weak_dirichlet", "strong_dirichlet",
            "flather", "free_slip"
        """
        if expression is None and bctype!="free_slip":
            raise TypeError("Boundary condition of type %s requires expression argument." %
                bctype)
        if expression is not None and bctype=="free_slip":
            raise TypeError('Boundary condition of type "free_slip" does not allow expression argument.')
        if facet_id is None:
            raise TypeError('facet_id argument to add_bc() method is not optional')

        self.append((function_name, expression, facet_id, bctype))

    def filter(self, function_name=None, bctype=None):
        """ Return a list of boundary conditions that satisfy the given
        criteria. """
        bcs = self

        if function_name is not None:
            bcs = [b for b in bcs if b[0] == function_name]

        if bctype is not None:
            bcs = [b for b in bcs if b[-1] == bctype]

        return BoundaryConditionSet(bcs)
