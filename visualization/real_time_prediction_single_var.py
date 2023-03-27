import argparse
import pathlib

import dash
import dash_bootstrap_components as dbc
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import skimage.transform

import onet_disk2D.grids
import onet_disk2D.model
import onet_disk2D.run


# todo add guidance
# todo add model infomation (scrollable window)
# tod add progress bar


def get_parser():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--args_file",
        type=str,
        default="args.yml",
        help="file that logs training args.",
    )
    parser.add_argument("--arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument("--fargo_setup_file", type=str, default="fargo_setups.yml")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory that store model files (params_struct.pkl, params.npy, etc). "
        "If empty, model_dir = run_dir. Use it for intermediate models in run_dir/xxx.",
    )

    # inputs
    parser.add_argument(
        "--num_cell_radial",
        type=int,
        help="Predicted image's radial resolution (NY in fargo3d).",
    )
    parser.add_argument(
        "--num_cell_azimuthal",
        type=int,
        help="Predicted image's azimuthal resolution (NX in fargo3d).",
    )

    return parser


class CartesianToPolar:
    def __init__(self, nxy, nr, ntheta, rmin, rmax, theta_min, theta_max):
        self.nxy = nxy
        self.nr = nr
        self.ntheta = ntheta
        self.rmin = rmin
        self.rmax = rmax
        self.theta_min = theta_min
        self.theta_max = theta_max

    def __call__(self, output_coor):
        # pixel to coordinate
        output_coor = 2 * self.rmax * (output_coor / self.nxy - 0.5)
        """(column, row)"""

        x, y = np.split(output_coor, 2, axis=1)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        """from -pi to pi"""

        # coordinate to pixel
        r = np.asarray(
            (r - self.rmin) / (self.rmax - self.rmin) * (self.nr - 1),
            dtype=int,
        )
        theta = np.asarray(
            (theta - self.theta_min)
            / (self.theta_max - self.theta_min)
            * (self.ntheta - 1),
            dtype=int,
        )

        return np.hstack([theta, r])


def sample_y(num_cell_radial, num_cell_azimuthal):
    ymin = float(job.fargo_setups["ymin"])
    ymax = float(job.fargo_setups["ymax"])
    ny = num_cell_radial if num_cell_radial else int(job.fargo_setups["ny"])
    nx = num_cell_azimuthal if num_cell_azimuthal else int(job.fargo_setups["nx"])
    # generate coords
    grids = onet_disk2D.grids.Grids(
        ymin=ymin, ymax=ymax, xmin=-np.pi, xmax=np.pi, ny=ny, nx=nx
    )
    return grids


def latex_float(f):
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str


def setup_slider(font_size=17):
    # todo improve the slider with https://twitter.com/plotlygraphs/status/1356259511330349060?s=20
    # todo slider ticks should be in LaTeX format
    # todo reduce number of ticks
    # todo add boundary
    # set parameter bars
    alpha_min = job.args["u_min"][0]
    alpha_max = job.args["u_max"][0]
    alpha_slider = dash.dcc.Slider(
        min=alpha_min,
        max=alpha_max,
        step=(alpha_max - alpha_min) / 100,
        marks={
            v: {"label": f"{10 ** v:.1e}", "style": {"font-size": font_size}}
            for v in np.linspace(alpha_min, alpha_max, 4)
        },
        value=-3,
        # tooltip={"placement": "top", "always_visible": True},
    )
    aspectratio_min = job.args["u_min"][1]
    aspectratio_max = job.args["u_max"][1]
    aspectratio_slider = dash.dcc.Slider(
        min=aspectratio_min,
        max=aspectratio_max,
        step=(aspectratio_max - aspectratio_min) / 100,
        marks={
            v: {"label": f"{v:.2g}", "style": {"font-size": font_size}}
            for v in np.linspace(aspectratio_min, aspectratio_max, 4)
        },
        value=0.05,
        # tooltip={"placement": "top", "always_visible": True},
    )
    planetmass_min = job.args["u_min"][2]
    planetmass_max = job.args["u_max"][2]
    mj = 9.548e-4
    planetmass_slider = dash.dcc.Slider(
        min=planetmass_min,
        max=planetmass_max,
        step=(planetmass_max - planetmass_min) / 100,
        marks={
            v: {"label": f"{10 ** v/mj:.1e}", "style": {"font-size": font_size}}
            for v in np.linspace(planetmass_min, planetmass_max, 4)
        },
        value=-3,
        # tooltip={"placement": "top", "always_visible": True},
    )

    return alpha_slider, aspectratio_slider, planetmass_slider


def update_graph(alpha, aspectratio, planetmass):
    """

    Args:
        alpha: in log10
        aspectratio: in linear scale
        planetmass: in log10

    Returns:

    """
    # todo colorbar ticks should be in LaTeX format
    alpha = 10.0**alpha
    planetmass = 10.0**planetmass
    u = jnp.array([alpha, aspectratio, planetmass])[None, :]
    r = y_grids.r_fargo_all[job.unknown_type]
    theta = y_grids.theta_fargo_all[job.unknown_type]

    inputs = {
        "u_net": u,
        "y_net": y_grids.coords_fargo_all[job.unknown_type].reshape((-1, 2)),
    }

    predict = job.s_pred_fn(job.model.params, job.state, inputs)
    predict = predict.reshape(y_grids.ny, y_grids.nx)
    nxy = 512
    cartesian_to_polar = CartesianToPolar(
        nxy=nxy,
        nr=len(r),
        ntheta=len(theta),
        rmin=min(r),
        rmax=max(r),
        theta_min=min(theta),
        theta_max=max(theta),
    )
    predict_in_cartesian = skimage.transform.warp(
        predict,
        cartesian_to_polar,
        output_shape=(nxy, nxy),
        order=3,
    )

    # vmin = float(np.nanmin(predict_in_cartesian))
    # vmax = float(np.nanmax(predict_in_cartesian))
    # a = 1.0 / (vmax - vmin)
    # b = -a * vmin
    # vmid = b
    vmin = -5
    vmax = 0.2
    a = 1.0 / (vmax - vmin)
    b = -a * vmin
    vmid = b

    # px.imshow
    x = y = np.linspace(-max(r), max(r), nxy)
    fig = px.imshow(
        predict_in_cartesian,
        x=x,
        y=y,
        range_color=[vmin, vmax],
        # aspect=1/3,
        origin="lower",
        aspect="equal",
    )
    r_lim = 1.2
    fig.update_xaxes(range=[-r_lim, r_lim])
    fig.update_yaxes(range=[-r_lim, r_lim], autorange=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        margin=dict(t=0, b=0),
    )
    # try to improve performance by disable imshow interactivity
    fig.update_traces(hovertemplate=None, hoverinfo="skip")
    fig.update_xaxes(
        {
            "title": {"text": "X", "standoff": 25, "font": {"size": 35}},
            # "scaleanchor": "y",
            # "scaleratio": y_grids.ny / y_grids.nx,
        }
    )
    fig.update_yaxes(
        {
            "title": {"text": "Y", "standoff": 25},
        }
    )
    # xaxes = {
    #     "tickmode": "array",
    #     "tickvals": [-np.pi / 2, 0.0, np.pi / 2],
    #     "ticktext": [
    #         r"$\huge{-\frac{1}{2}\pi}$",
    #         r"$\huge{0}$",
    #         r"$\huge{\frac{1}{2}\pi}$",
    #     ],
    # }
    # fig.update_xaxes(xaxes)
    # if np.ceil(vmax) - np.ceil(vmin) < 3:
    #     colorbar_ticks = np.linspace(np.round(vmin, 2), np.round(vmax, 2), 4, endpoint=False)
    #     fig.update_coloraxes(
    #         {
    #             "colorscale": [(0.0, "blue"), (vmid, "white"), (1.0, "red")],
    #             "colorbar": {
    #                 "tickvals": colorbar_ticks,
    #                 "ticktext": [f"{10 ** v:.1f}" for v in colorbar_ticks],
    #                 "len": 0.8,
    #             },
    #         }
    #     )
    # else:
    #     colorbar_ticks = np.arange(np.ceil(vmin), np.ceil(vmax), 1)
    #     # layout
    #     fig.update_coloraxes(
    #         {
    #             "colorscale": [(0.0, "blue"), (vmid, "white"), (1.0, "red")],
    #             "colorbar": {
    #                 "tickvals": colorbar_ticks,
    #                 "ticktext": [f"{10**v:.0e}" for v in colorbar_ticks],
    #                 "len": 0.8,
    #             },
    #         }
    #     )
    colorbar_ticks = [-5, -4, -3, -2, -1, vmax]
    # layout
    fig.update_coloraxes(
        {
            "colorscale": [(0.0, "blue"), (vmid, "white"), (1.0, "red")],
            "colorbar": {
                "tickvals": colorbar_ticks,
                "ticktext": [
                    # f"{10**v:.0e}" if v < -2 else f"{10**v:.1f}" for v in colorbar_ticks
                    "10e-5",
                    "10e-4",
                    "0.001",
                    "0.01",
                    "0.1",
                    f"{10**vmax:.1f}",
                ],
                "len": 0.8,
            },
        }
    )

    fig.update_layout({"font": {"size": 30, "family": "Times New Roman"}})

    return fig


def update_alpha(alpha):
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


def update_aspectratio(aspectratio):
    """

    Args:
        aspectratio: in linear scale

    Returns:

    """
    return r"Scale height ($h_0$): " + f"{aspectratio:.3f}"


def update_planetmass(planetmass):
    """

    Args:
        planetmass: in log10

    Returns:

    """
    # jupiter mass to solar mass ratio
    mj = 9.548e-4
    planetmass = f"{10.0**planetmass/mj:.3f}"
    return r"Planet mass ($M_p$): " + planetmass + " $M_J$"


if __name__ == "__main__":
    predict_args = get_parser().parse_args()
    run_dir = pathlib.Path(predict_args.run_dir).resolve()
    if predict_args.model_dir:
        model_dir = pathlib.Path(predict_args.model_dir).resolve()
    else:
        model_dir = run_dir

    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        predict_args.args_file,
        predict_args.arg_groups_file,
        predict_args.fargo_setup_file,
    )

    job = onet_disk2D.run.JOB(job_args)
    job.load_model(model_dir)

    y_grids = sample_y(predict_args.num_cell_radial, predict_args.num_cell_azimuthal)

    app = dash.Dash(
        __name__,
        # external_scripts=[
        #     "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
        # ],
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="PPDONet",
    )
    server = app.server
    alpha_slider, aspectratio_slider, planetmass_slider = setup_slider(font_size=20)
    # set graph
    graph = dash.dcc.Graph(
        responsive=True,
        mathjax=True,
        style={
            "aspect-ratio": "1/1",
            "width": "80vh",
            "marginLeft": "auto",
            "marginRight": "auto"
            # "height": "100vh",
        },
    )
    # set layout
    header_row = dbc.Row(
        [
            dbc.Col(
                [
                    dash.html.H1(
                        "DeepONet predictions of steady protoplanetary disks' gas density",
                    ),
                    dash.html.P(
                        [
                            "Shunyuan Mao  ",
                            dash.dcc.Link(
                                "Source code",
                                href="https://github.com/smao-astro/PPDONet",
                                target="_blank",
                            )
                            # add paper link here
                        ],
                        style={"font-size": "30px"},
                    ),
                ],
                align="end",
            )
        ],
        # style={"height": "10%"},
    )
    alpha_text = dash.dcc.Markdown(mathjax=True)
    aspectratio_text = dash.dcc.Markdown(mathjax=True)
    planetmass_text = dash.dcc.Markdown(mathjax=True)

    slider_stype = {
        "marginBottom": "20px",
        "marginLeft": "20px",
        "marginRight": "10px",
    }
    content_row = dbc.Row(
        [
            dbc.Col(
                [
                    alpha_text,
                    dash.html.Div(alpha_slider, style=slider_stype),
                    aspectratio_text,
                    dash.html.Div(aspectratio_slider, style=slider_stype),
                    planetmass_text,
                    dash.html.Div(planetmass_slider, style=slider_stype),
                ],
                align="start",
                width=3,
                style={"font-size": "30px"},
            ),  # bar
            # dbc.Col(dbc.Spinner(graph), width=True, align="center"),  # graph
            dbc.Col(graph, width=True, align="center"),  # graph
        ],
        justify="center",
        # style={"height": "60%"},
    )
    app.layout = dbc.Container(
        [header_row, dash.html.Hr(), content_row],  # header title
        # style={"height": "90vh"},
        fluid=True,
    )
    # connect graph to bar
    app.callback(
        dash.Output(graph, component_property="figure"),
        dash.Input(alpha_slider, component_property="value"),
        dash.Input(aspectratio_slider, component_property="value"),
        dash.Input(planetmass_slider, component_property="value"),
    )(update_graph)

    app.callback(
        dash.Output(alpha_text, component_property="children"),
        dash.Input(alpha_slider, component_property="value"),
    )(update_alpha)

    app.callback(
        dash.Output(aspectratio_text, component_property="children"),
        dash.Input(aspectratio_slider, component_property="value"),
    )(update_aspectratio)

    app.callback(
        dash.Output(planetmass_text, component_property="children"),
        dash.Input(planetmass_slider, component_property="value"),
    )(update_planetmass)

    # run server
    # app.run(debug=True)
    app.run(debug=False)
