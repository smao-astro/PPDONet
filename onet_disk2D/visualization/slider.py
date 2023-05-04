import dash
import numpy as np


def update_alpha_text(alpha):
    """

    Args:
        alpha: in log10

    Returns:

    """
    alpha = f"{10.0**alpha:.2e}"
    if "e" in alpha:
        base, exponent = alpha.split("e")
        alpha = r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    return r"Alpha viscosity ($\alpha$): " + alpha


def setup_alpha_slider(vmin, vmax, value_init=-3, class_name="mb-3"):
    """

    Args:
        vmin: in log10
        vmax: in log10
        value_init:
        class_name:

    Returns:

    """
    slider = dash.dcc.Slider(
        min=vmin,
        max=vmax,
        marks={
            v: {
                "label": f"{10 ** v:.1g}",
                # "style": {"font-size": font_size}
            }
            # the stop parameter is alpha_max, but it won't be included in the slider range for whatever reason, minus 1e-4 to include it
            for v in np.linspace(vmin, vmax - 1e-4, 4)
        },
        value=value_init,
        className=class_name,
    )
    return slider


def update_aspectratio_text(aspectratio):
    """

    Args:
        aspectratio: in linear scale

    Returns:

    """
    return r"Scale height ($h_0$): " + f"{aspectratio:.3f}"


def setup_aspectratio_slider(vmin, vmax, value_init=0.05, class_name="mb-3"):
    """

    Args:
        vmin: in linear
        vmax: in linear
        value_init:
        class_name:

    Returns:

    """
    slider = dash.dcc.Slider(
        min=vmin,
        max=vmax,
        marks={
            v: {
                "label": f"{v:.1g}",
                # "style": {"font-size": font_size}
            }
            for v in np.linspace(vmin, vmax, 4)
        },
        value=value_init,
        className=class_name,
    )
    return slider


def update_planetmass_text(planetmass):
    """

    Args:
        planetmass: in log10

    Returns:

    """
    # jupiter mass to solar mass ratio
    mj = 9.548e-4
    planetmass = f"{10.0**planetmass/mj:.3f}"
    return r"Planet mass ($M_p$): " + planetmass + " $M_J$"


def setup_planetmass_slider(vmin, vmax, value_init=-3, class_name="mb-3"):
    """

    Args:
        vmin: in log10
        vmax: in log10
        value_init:
        class_name:

    Returns:

    """
    mj = 9.548e-4
    slider = dash.dcc.Slider(
        min=vmin,
        max=vmax,
        marks={
            v: {
                "label": f"{10 ** v/mj:.1g}",
                # "style": {"font-size": font_size}
            }
            for v in np.linspace(vmin, vmax, 4)
        },
        value=value_init,
        className=class_name,
    )
    return slider
