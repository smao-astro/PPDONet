"""
To optimize the computational cost, we can:
1) reduce the grid resolution (easy to do)
2) convert the data to a figure file (RGB array?), and then display it.
"""
import argparse
import pathlib

# add a timer for import
import time

import dash
import dash_bootstrap_components as dbc
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import onet_disk2D.grids
import onet_disk2D.model
import onet_disk2D.run
import onet_disk2D.visualization.slider
from onet_disk2D.utils import timer
from onet_disk2D.visualization.utils import mpl_to_uri

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
        "--nxy",
        type=int,
        default=768,
        help="Cartesian image's resolution.",
    )

    return parser


@timer
def setup_slider():
    # todo improve the slider with https://twitter.com/plotlygraphs/status/1356259511330349060?s=20
    # todo add boundary
    # set parameter bars
    alpha_min = job.args["u_min"][0]
    alpha_max = job.args["u_max"][0]
    alpha_slider = dash.dcc.Slider(
        min=alpha_min,
        max=alpha_max,
        marks={
            v: {
                "label": f"{10 ** v:.1g}",
                # "style": {"font-size": font_size}
            }
            # the stop parameter is alpha_max, but it won't be included in the slider range for whatever reason, minus 1e-4 to include it
            for v in np.linspace(alpha_min, alpha_max - 1e-4, 4)
        },
        value=-3,
        className="mb-3",
        # tooltip={"placement": "top", "always_visible": True},
    )
    aspectratio_min = job.args["u_min"][1]
    aspectratio_max = job.args["u_max"][1]
    aspectratio_slider = dash.dcc.Slider(
        min=aspectratio_min,
        max=aspectratio_max,
        marks={
            v: {
                "label": f"{v:.1g}",
                # "style": {"font-size": font_size}
            }
            for v in np.linspace(aspectratio_min, aspectratio_max, 4)
        },
        value=0.05,
        className="mb-3",
        # tooltip={"placement": "top", "always_visible": True},
    )
    planetmass_min = job.args["u_min"][2]
    planetmass_max = job.args["u_max"][2]
    mj = 9.548e-4
    planetmass_slider = dash.dcc.Slider(
        min=planetmass_min,
        max=planetmass_max,
        marks={
            v: {
                "label": f"{10 ** v/mj:.1g}",
                # "style": {"font-size": font_size}
            }
            for v in np.linspace(planetmass_min, planetmass_max, 4)
        },
        value=-3,
        className="mb-3",
        # tooltip={"placement": "top", "always_visible": True},
    )

    return alpha_slider, aspectratio_slider, planetmass_slider


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
        job,
        nxy,
        vmin=-2,
        vmax=0.2,
        vcenter=0,
        plot_limit: float = 2.0,
    ):
        self.job = job
        self.nxy = nxy
        self.vmin = vmin
        self.vmax = vmax
        self.vcenter = vcenter
        self.r_min = float(job.fargo_setups["ymin"])
        self.r_max = float(job.fargo_setups["ymax"])
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

        # colorscale
        self.norm = CustomNormalize(self.vmin, self.vmax)
        self.colormap = matplotlib.colormaps["jet"]
        # colorbar
        self.colorbar_ticks = np.array([-2, -1, 0, self.vmax / 2.0, self.vmax])
        self.colorbar_ticktext = [
            "0.01",
            "0.1",
            "1",
            f"{10**(self.vmax/2.):.1f}",
            f"{10**self.vmax:.1f}",
        ]

    @timer
    def predict(self, alpha, aspectratio, planetmass):
        alpha = 10.0**alpha
        planetmass = 10.0**planetmass
        u = jnp.array([alpha, aspectratio, planetmass])[None, :]
        inputs = {"u_net": u, "y_net": self.y_net}
        predict = self.job.s_pred_fn(self.job.model.params, self.job.state, inputs)
        # reshape
        predict = predict.reshape(self.nxy, self.nxy)
        # normalize by initial condition
        predict = predict + 0.5 * np.log10(self.r)
        # mask
        predict = np.where(self.r_mask, predict, np.nan)
        return predict

    @timer
    def update(self, alpha, aspectratio, planetmass):
        predict = self.predict(alpha, aspectratio, planetmass)

        fig = plt.figure(layout="constrained")
        plt.imshow(
            predict,
            cmap=self.colormap,
            norm=self.norm,
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
        plt.title("Predicted normalized surface density", fontsize=BIGGER_SIZE)
        cbar = plt.colorbar(ticks=self.colorbar_ticks)
        cbar.ax.set_yticklabels(self.colorbar_ticktext)

        # Convert the Matplotlib figure to an image URI and return it
        return mpl_to_uri(fig)


@timer
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


@timer
def update_aspectratio(aspectratio):
    """

    Args:
        aspectratio: in linear scale

    Returns:

    """
    return r"Scale height ($h_0$): " + f"{aspectratio:.3f}"


@timer
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


@timer
def load_model(predict_args):
    """Load model from checkpoint.

    Args:
        predict_args: parsed arguments from command line

    """
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

    return job


if __name__ == "__main__":
    predict_args = get_parser().parse_args()
else:
    # if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    # running on gunicorn server (Heroku)
    predict_args = get_parser().parse_args(
        [
            "--run_dir",
            "trained_network/single_log_sigma",
            "--num_cell_radial",
            "200",
            "--num_cell_azimuthal",
            "600",
        ]
    )

job = load_model(predict_args)

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

modal_content = dash.dcc.Markdown(
    r"""
    ## Instructions & Help

    1. **Introduction**:
    Welcome to my PPDONet (Protoplanetary Disk Operator Network) demo app! This app is designed to provide you with real-time neural network predictions of disk-planet systems using your input disk and planet parameters.
    
    2. **How to Use**:
       - *Step 1*: Move the sliders to adjust the disk and planet parameters.
       - *Step 2*: View the surface density image generated by the neural network.
       - *Step 3 \[Optional\]*: Click the "Source Code" button to navigate to the GitHub repository for this app.
    
    3. **Understanding the Image**:
    The predicted image represents the quasi-steady-state surface density distribution of the disk-planet system with the disk-planet parameters on the left. The colorbar on the right shows the surface density normalized by the initial surface density at the planet's radius $(1.0, 0.0)$.
    
    4. **About the Neural Network**:
    We use a modified Deep Operator Network (DeepONet, [Lu et al. 2021](https://www.nature.com/articles/s42256-021-00302-5)) to predict the disk-planet system. The network is trained on more than 400 FARGO3D simulations with randomly generated disk and planet parameters. The trained network takes less than 1s to predict the image on a laptop.
    
    For more detail, please wait for our paper, which is currently under review.
    
    For any additional questions or support, please feel free to contact us at Shunyuan Mao (symao@uvic.ca). Enjoy exploring the app!
    """,
    mathjax=True,
)
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
alpha_slider, aspectratio_slider, planetmass_slider = setup_slider()
left_column = [
    alpha_text,
    alpha_slider,
    aspectratio_text,
    aspectratio_slider,
    planetmass_text,
    planetmass_slider,
]
left_column = dbc.Card(
    [
        dbc.CardHeader("Input Parameters", className="fs-3"),
        dbc.CardBody(
            left_column, className="d-flex flex-column justify-content-evenly"
        ),
    ],
    # prevent left column and right column overlap on small screen
    className="text-body mb-4 mb-lg-0",
)
# set graph
my_graph = Graph(
    job,
    nxy=predict_args.nxy,
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
            left_column,
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
)(my_graph.update)

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

if __name__ == "__main__":
    # Warning: the lines below should not be executed when running on Heroku/PythonAnywhere
    # See https://help.pythonanywhere.com/pages/Flask/#do-not-use-apprun
    # run server
    # app.run(debug=True)
    app.run(debug=False, port=8052)

# end timer
print(f"Initialize app takes {time.perf_counter() - start:.2f} seconds")
