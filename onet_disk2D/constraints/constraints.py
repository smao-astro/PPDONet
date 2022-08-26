import jaxphyinf.constraints


class ParametricConstraints(jaxphyinf.constraints.Constraints):
    """Constraints with parameters as one of the two inputs

    Attributes:
        parameters: The structure of parameters should consistent with `self.samples`.

    """

    def __init__(self, *args, **kwargs):
        super(ParametricConstraints, self).__init__(*args, **kwargs)
        self.parameters = {}

    def get_v_g(self, *args):
        """Compute loss values and gradients.

        Args:
            *args: (params,) or (params, state)

        Returns:
            (dict, dict)
        """
        v_g = {
            k: v_g_fn(*args, self.parameters[k], self.samples[k])
            for k, v_g_fn in self.v_g_fn.items()
        }
        return {k: v[0] for k, v in v_g.items()}, {k: v[1] for k, v in v_g.items()}
