"""
To optimize the computational cost, we can:
1) reduce the grid resolution (easy to do)
2) convert the data to a figure file (RGB array?), and then display it.
"""
import argparse
import functools
import importlib.resources as pkg_resources
import pathlib
import time
from typing import Mapping

import astropy.io.fits as fits_module
import dash
import dash_bootstrap_components as dbc
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt

import onet_disk2D.grids
import onet_disk2D.model
import onet_disk2D.run
import onet_disk2D.visualization
from onet_disk2D.utils import timer

# If you don't need to display the plot interactively, you can use the Agg backend
matplotlib.use("Agg")

# deprecated in later version of jax
# jax.config.update('jax_platforms_name', 'cpu')
jax.config.update("jax_platforms", "cpu")

# todo display software version

# matplotlib config
SMALL_SIZE = 8 * 1.5
MEDIUM_SIZE = 10 * 1.5
BIGGER_SIZE = 12 * 1.5

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)
# font family
plt.rcParams["font.family"] = "Times New Roman"


def get_parser():
    parser = argparse.ArgumentParser()
    # IO
    ## sigma
    parser.add_argument("--sigma_run_dir", type=str, required=True)
    parser.add_argument(
        "--sigma_args_file",
        type=str,
        default="args.yml",
        help="file that logs training args.",
    )
    parser.add_argument("--sigma_arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument(
        "--sigma_fargo_setup_file", type=str, default="fargo_setups.yml"
    )
    parser.add_argument(
        "--sigma_model_dir",
        type=str,
        default="",
        help="Directory that store model files (params_struct.pkl, params.npy, etc). "
        "If empty, model_dir = run_dir. Use it for intermediate models in run_dir/xxx.",
    )
    ## v_r
    parser.add_argument("--v_r_run_dir", type=str, required=True)
    parser.add_argument(
        "--v_r_args_file",
        type=str,
        default="args.yml",
        help="file that logs training args.",
    )
    parser.add_argument("--v_r_arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument("--v_r_fargo_setup_file", type=str, default="fargo_setups.yml")
    parser.add_argument(
        "--v_r_model_dir",
        type=str,
        default="",
        help="Directory that store model files (params_struct.pkl, params.npy, etc). "
        "If empty, model_dir = run_dir. Use it for intermediate models in run_dir/xxx.",
    )
    ## v_theta
    parser.add_argument("--v_theta_run_dir", type=str, required=True)
    parser.add_argument(
        "--v_theta_args_file",
        type=str,
        default="args.yml",
        help="file that logs training args.",
    )
    parser.add_argument("--v_theta_arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument(
        "--v_theta_fargo_setup_file", type=str, default="fargo_setups.yml"
    )
    parser.add_argument(
        "--v_theta_model_dir",
        type=str,
        default="",
        help="Directory that store model files (params_struct.pkl, params.npy, etc). "
        "If empty, model_dir = run_dir. Use it for intermediate models in run_dir/xxx.",
    )
    # inputs
    parser.add_argument(
        "--nxy",
        type=int,
        default=768,
        help="Cartesian image's resolution.",
    )

    return parser


class CustomNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax):
        super().__init__(vmin=vmin, vmax=vmax)

    def __call__(self, value, clip=None):
        value = np.asarray(value)
        result = np.empty_like(value)
        positive = value >= 0
        negative = ~positive

        # result[positive] = 0.5 * (1.0 + np.sqrt(value[positive] / self.vmax))
        # Positive part: map [0, vmax] linearly to [0.5, 1]
        result[positive] = 0.5 + 0.5 * value[positive] / self.vmax

        # Negative part: map [vmin, 0] to [0, 0.5], with denser spacing near 0
        negative_values = value[negative]
        result[negative] = 0.5 * (1 - np.sqrt(negative_values / self.vmin))

        return np.ma.array(result, mask=np.ma.getmask(value))

    def inverse(self, value):
        value = np.asarray(value)
        result = np.empty_like(value)
        positive = value >= 0.5
        negative = ~positive

        # Inverse for positive part
        # result[positive] = (2.0 * value[positive] - 1.0) ** 2 * self.vmax
        result[positive] = (value[positive] - 0.5) * 2 * self.vmax

        # Inverse for negative part
        negative_values = value[negative]
        result[negative] = self.vmin * (1 - 2 * negative_values) ** 2

        return np.ma.array(result, mask=np.ma.getmask(value))


class Graph:
    def __init__(
        self,
        predict_args,
        nxy: int,
        r_min: float,
        r_max: float,
        vmin: Mapping[str, float],
        vmax: Mapping[str, float],
        plot_limit: float = 2.0,
    ):
        self.predict_args = predict_args
        self.nxy = nxy
        self.vmin = vmin
        self.vmax = vmax
        self.r_min = r_min
        self.r_max = r_max
        self.xy_limit = plot_limit

        self.x = self.y = np.linspace(-self.xy_limit, self.xy_limit, self.nxy)
        x_grid, y_grid = np.meshgrid(self.x, self.y, indexing="xy")
        """x_grid and y_grid are 2D arrays of shape (nxy, nxy)"""
        self.r = np.sqrt(x_grid**2 + y_grid**2)
        """r is a 2D array of shape (nxy, nxy)"""
        self.r_mask = np.logical_and(self.r >= self.r_min, self.r <= self.r_max)
        self.theta = np.arctan2(y_grid, x_grid)
        """theta is a 2D array of shape (nxy, nxy)"""
        self.y_net = np.stack([self.r, self.theta], axis=-1).reshape((-1, 2))
        """y_net is a 2D array of shape (nxy**2, 2)"""

    @functools.cached_property
    def norm(self):
        return {
            "sigma": CustomNormalize(vmin=self.vmin["sigma"], vmax=self.vmax["sigma"]),
            "v_r": matplotlib.colors.SymLogNorm(
                linthresh=0.03,
                linscale=0.1,
                vmin=self.vmin["v_r"],
                vmax=self.vmax["v_r"],
            ),
            "v_theta": matplotlib.colors.SymLogNorm(
                linthresh=0.03,
                linscale=0.1,
                vmin=self.vmin["v_theta"],
                vmax=self.vmax["v_theta"],
            ),
        }

    @functools.cached_property
    def colormap(self):
        return {
            "sigma": matplotlib.colormaps["jet"],
            "v_r": matplotlib.colormaps["RdBu_r"],
            "v_theta": matplotlib.colormaps["RdBu_r"],
        }

    @functools.cached_property
    def colorbar_ticks(self):
        return {
            "sigma": np.array(
                [-2, -1, 0, self.vmax["sigma"] / 2.0, self.vmax["sigma"]]
            ),
            "v_r": np.array([-1, -0.5, 0, 0.5, 1]),
            "v_theta": np.array([-1, -0.5, 0, 0.5, 1]),
        }

    @functools.cached_property
    def colorbar_ticktext(self):
        return {
            "sigma": [
                "0.01",
                "0.1",
                "1",
                f"{10**(self.vmax['sigma']/2.):.1f}",
                f"{10**self.vmax['sigma']:.1f}",
            ],
            "v_r": ["-1", "-0.5", "0", "0.5", "1"],
            "v_theta": ["-1", "-0.5", "0", "0.5", "1"],
        }

    @functools.cached_property
    def fig_title(self):
        return {
            "sigma": "Predicted normalized surface density",
            "v_r": "Predicted perturbed radial velocity",
            "v_theta": "Predicted perturbed azimuthal velocity",
        }

    @timer
    def predict(self, alpha, aspectratio, planetmass, phy_variable):
        alpha = 10.0**alpha
        planetmass = 10.0**planetmass
        u = jnp.array([alpha, aspectratio, planetmass])[None, :]
        inputs = {"u_net": u, "y_net": self.y_net}
        job = load_model(
            getattr(predict_args, phy_variable + "_run_dir"),
            getattr(predict_args, phy_variable + "_args_file"),
            getattr(predict_args, phy_variable + "_arg_groups_file"),
            getattr(predict_args, phy_variable + "_fargo_setup_file"),
            getattr(predict_args, phy_variable + "_model_dir"),
        )
        predict = job.s_pred_fn(job.model.params, job.state, inputs)
        # reshape
        predict = predict.reshape(self.nxy, self.nxy)
        if phy_variable == "sigma":
            # normalize by initial condition
            predict = predict + 0.5 * np.log10(self.r)
        elif phy_variable == "v_r":
            cs = aspectratio
            # remove initial condition (background) to get perturbed v_r
            predict -= (
                -1.5 * alpha * aspectratio**2 * np.sqrt((1.0 + planetmass) / self.r)
            )
            # normalize by sound speed
            predict = predict / cs
        elif phy_variable == "v_theta":
            cs = aspectratio
            # convert to non-rotating frame
            predict += self.r
            # remove initial condition (background) to get perturbed v_theta
            predict -= np.sqrt(1 - 1.5 * aspectratio**2) * np.sqrt(
                (1.0 + planetmass) / self.r
            )
            # normalize by sound speed
            predict = predict / cs
        else:
            raise ValueError(f"Unknown phy_variable: {phy_variable}")
        # mask
        predict = np.where(self.r_mask, predict, np.nan)
        return predict

    @timer
    def update(self, alpha, aspectratio, planetmass, phy_variable):
        predict = self.predict(alpha, aspectratio, planetmass, phy_variable)

        fig = plt.figure(layout="constrained")
        plt.imshow(
            predict,
            cmap=self.colormap[phy_variable],
            norm=self.norm[phy_variable],
            aspect="equal",
            origin="lower",
            extent=(
                -self.xy_limit,
                self.xy_limit,
                -self.xy_limit,
                self.xy_limit,
            ),
        )
        plt.xlabel("X (Planet Radius)")
        plt.ylabel("Y (Planet Radius)")
        plt.title(self.fig_title[phy_variable], fontsize=BIGGER_SIZE)
        cbar = plt.colorbar(ticks=self.colorbar_ticks[phy_variable])
        cbar.ax.set_yticklabels(self.colorbar_ticktext[phy_variable])

        # Convert the Matplotlib figure to an image URI and return it
        return onet_disk2D.visualization.mpl_to_uri(fig)

    def write_fits(self, bytes_io, alpha, aspectratio, planetmass, phy_variable):
        predict = self.predict(alpha, aspectratio, planetmass, phy_variable)
        primary_hdu = fits_module.PrimaryHDU()
        # write some info to the header
        primary_hdu.header["ALPHA"] = (10**alpha, "Alpha viscosity")
        primary_hdu.header["H0"] = (
            aspectratio,
            "Disk aspect ratio, constant in the disk",
        )
        primary_hdu.header["PMASS"] = (10**planetmass, "Planet-to-star mass ratio")
        fits_data = fits_module.HDUList(
            [primary_hdu, fits_module.ImageHDU(data=predict)]
        )
        fits_data.writeto(bytes_io)

    def download(self, alpha, aspectratio, planetmass, phy_variable, n_clicks):
        return dash.dcc.send_bytes(
            src=self.write_fits,
            filename=f"alpha_{alpha:.2e}_h0_{aspectratio:.2f}_q_{planetmass:.2e}_{phy_variable}.fits",
            alpha=alpha,
            aspectratio=aspectratio,
            planetmass=planetmass,
            phy_variable=phy_variable,
        )


@timer
def load_model(run_dir, args_file, arg_groups_file, fargo_setup_file, model_dir):
    """Load model from checkpoint."""
    run_dir = pathlib.Path(run_dir).resolve()
    if model_dir:
        model_dir = pathlib.Path(model_dir).resolve()
    else:
        model_dir = run_dir

    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        args_file,
        arg_groups_file,
        fargo_setup_file,
    )

    job = onet_disk2D.run.JOB(job_args)
    job.load_model(model_dir)

    return job


if __name__ == "__main__":
    predict_args = get_parser().parse_args()
else:
    # if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    # running on gunicorn server (Heroku)
    predict_args = get_parser().parse_args(
        [
            "--sigma_run_dir",
            "trained_network/single_log_sigma",
            "--v_r_run_dir",
            "trained_network/single_v_r",
            "--v_theta_run_dir",
            "trained_network/single_v_theta",
        ]
    )

# jobs = {
#     k: load_model(
#         getattr(predict_args, k + "_run_dir"),
#         getattr(predict_args, k + "_args_file"),
#         getattr(predict_args, k + "_arg_groups_file"),
#         getattr(predict_args, k + "_fargo_setup_file"),
#         getattr(predict_args, k + "_model_dir"),
#     )
#     for k in ["sigma", "v_r", "v_theta"]
# }

# add a timer for initialize app
start = time.perf_counter()

app = dash.Dash(
    __name__,
    # external_scripts=[
    #     "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
    # ],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="PPDONet",
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=0.5",
        },
    ],
)
server = app.server

# todo update the readme
with pkg_resources.open_text(
    "onet_disk2D.visualization", "real_time_prediction_readme.md"
) as f:
    readme = f.read()
modal_content = dash.dcc.Markdown(readme, mathjax=True)

open_button = dbc.Button("Instructions & Help", n_clicks=0)
close_button = dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Instructions & Help")),
        dbc.ModalBody(modal_content),
        dbc.ModalFooter(close_button),
    ],
    is_open=True,
)

# set layout
header_row = [
    dash.html.H1("PPDONet", className="text-center text-primary"),
    dash.html.H2(
        "Deep Operator Networks for Fast Prediction of Steady-State Solutions in Disk-Planet Systems",
        className="text-center text-secondary",
    ),
    dash.html.H3(
        "Shunyuan Mao (symao@uvic.ca)",
        className="text-center text-body",
    ),
    dash.html.H3(
        [
            open_button,
            modal,
            dash.dcc.Link(
                "Source code",
                href="https://github.com/smao-astro/PPDONet",
                target="_blank",
            ),
        ],
        className="d-flex justify-content-between flex-nowrap",
    ),
]
header_row = dbc.Row(
    [dbc.Col(header_row)],
)
alpha_text = dash.dcc.Markdown(mathjax=True)
aspectratio_text = dash.dcc.Markdown(mathjax=True)
planetmass_text = dash.dcc.Markdown(mathjax=True)

# ===========
# set up the sliders
# check if the u_min and u_max are the same for all jobs

with (pathlib.Path(predict_args.sigma_run_dir) / predict_args.sigma_args_file).open(
    "r"
) as f:
    sigma_args = yaml.safe_load(f)
    u_min = sigma_args["u_min"]
    u_max = sigma_args["u_max"]
with (
    pathlib.Path(predict_args.sigma_run_dir) / predict_args.sigma_fargo_setup_file
).open("r") as f:
    fargo_setups = yaml.safe_load(f)
    r_min = float(fargo_setups["YMIN"])
    r_max = float(fargo_setups["YMAX"])

alpha_slider = onet_disk2D.visualization.setup_alpha_slider(u_min[0], u_max[0])
aspectratio_slider = onet_disk2D.visualization.setup_aspectratio_slider(
    u_min[1], u_max[1]
)
planetmass_slider = onet_disk2D.visualization.setup_planetmass_slider(
    u_min[2], u_max[2]
)
# ===========

# ===========
# set up the left column
left_column = [
    alpha_text,
    alpha_slider,
    aspectratio_text,
    aspectratio_slider,
    planetmass_text,
    planetmass_slider,
]
sliders = dbc.Card(
    [
        dbc.CardHeader("Input Parameters", className="fs-3"),
        dbc.CardBody(
            left_column, className="d-flex flex-column justify-content-evenly"
        ),
    ],
    # prevent left column and right column overlap on small screen
    className="text-body mb-4 mb-lg-2",
)
# ===========

# ===========
# setup the dropdown
dropdown = dash.dcc.Dropdown(
    options=[
        {"label": "Surface density", "value": "sigma"},
        {"label": "Radial velocity", "value": "v_r"},
        {"label": "Azimuthal velocity", "value": "v_theta"},
    ],
    value="sigma",
)
dropdown_card = dbc.Card(
    [
        dbc.CardHeader("Select quantity to view", className="fs-3"),
        dbc.CardBody(dropdown),
    ],
    className="text-body mb-3 mb-lg-2",
)
# ===========
download_button = dbc.Button("Download FITS", color="primary", size="lg")
download = dash.dcc.Download()

vmin = {
    "sigma": -2,
    "v_r": -1.0,
    "v_theta": -1.0,
}
vmax = {
    "sigma": 0.2,
    "v_r": 1.0,
    "v_theta": 1.0,
}
# set graph
my_graph = Graph(
    predict_args, nxy=predict_args.nxy, r_min=r_min, r_max=r_max, vmin=vmin, vmax=vmax
)
graph = dash.html.Img(
    style={
        "width": "100%",
        "height": "auto",
    },
    className="shadow",
)
content_row = dbc.Row(
    [
        dbc.Col(
            [
                sliders,
                dropdown_card,
                dash.html.Div([download_button], className="d-grid gap-2 mb-3 mb-lg-0"),
                download,
            ],
            xs=10,
            sm=10,
            md=10,
            lg=4,
            xl=4,
            className="fs-4",  # font size
        ),
        # dbc.Col(dbc.Spinner(graph), width=True, align="center"),  # graph
        dbc.Col(
            graph,
            className="d-flex justify-content-center align-items-center",
            xs=10,
            sm=10,
            md=10,
            lg=8,
            xl=8,
        ),
    ],
    justify="center",
)
app.layout = dbc.Container(
    [header_row, dash.html.Hr(), content_row],
    className="app-container",
    fluid=True,
)


@app.callback(
    dash.Output(modal, "is_open"),
    [dash.Input(open_button, "n_clicks"), dash.Input(close_button, "n_clicks")],
    [dash.State(modal, "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# connect graph to bar

app.callback(
    dash.Output(graph, component_property="src"),
    dash.Input(alpha_slider, component_property="value"),
    dash.Input(aspectratio_slider, component_property="value"),
    dash.Input(planetmass_slider, component_property="value"),
    dash.Input(dropdown, component_property="value"),
)(my_graph.update)

app.callback(
    dash.Output(alpha_text, component_property="children"),
    dash.Input(alpha_slider, component_property="value"),
)(onet_disk2D.visualization.update_alpha_text)

app.callback(
    dash.Output(aspectratio_text, component_property="children"),
    dash.Input(aspectratio_slider, component_property="value"),
)(onet_disk2D.visualization.update_aspectratio_text)

app.callback(
    dash.Output(planetmass_text, component_property="children"),
    dash.Input(planetmass_slider, component_property="value"),
)(onet_disk2D.visualization.update_planetmass_text)

app.callback(
    dash.Output(download, "data"),
    dash.State(alpha_slider, component_property="value"),
    dash.State(aspectratio_slider, component_property="value"),
    dash.State(planetmass_slider, component_property="value"),
    dash.State(dropdown, component_property="value"),
    dash.Input(download_button, "n_clicks"),
    prevent_initial_call=True,
)(my_graph.download)

if __name__ == "__main__":
    # Warning: the lines below should not be executed when running on Heroku/PythonAnywhere
    # See https://help.pythonanywhere.com/pages/Flask/#do-not-use-apprun
    # run server
    # app.run(debug=True)
    app.run(debug=False, port=8052)

# end timer
print(f"Initialize app takes {time.perf_counter() - start:.2f} seconds")
