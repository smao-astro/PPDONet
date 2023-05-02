import argparse
import functools
import pathlib

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import xarray as xr
import yaml

import onet_disk2D.callbacks
import onet_disk2D.data
import onet_disk2D.gradients
import onet_disk2D.grids
import onet_disk2D.model
import onet_disk2D.physics
import onet_disk2D.utils


def resolve_save_dir(save_dir, file_list, verbose=True):
    """Resolve save_dir from file_list.

    This is necessary for storing the results to a parent guild run.
    """
    save_dir = pathlib.Path(save_dir)
    # resolve soft links
    for file in file_list:
        if (save_dir / file).exists():
            save_dir = (save_dir / file).resolve().parent
            break
    else:
        raise FileNotFoundError(f"Can not find {file_list} in {save_dir}")

    if verbose:
        print(f"save_dir={save_dir}")
    return save_dir


def setup_save_dir(save_dir, model_dir):
    """Setup save_dir.

    If save_dir is None or empty string, save to model_dir or the parent guild run.
    """
    # save_dir
    if save_dir:
        save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir()
    else:
        # save to train dir
        save_dir = resolve_save_dir(
            model_dir, ["params.npy", "params_struct.pkl"], verbose=False
        )

    print(f"save_dir={save_dir}")

    return save_dir


def load_fargo_setups(fargo_setups_file):
    """Load fargo setups from fargo_setups_file.

    Args:
        fargo_setups_file:

    Returns:
        fargo_setups: dict of fargo setups, all keys are in lowercase, all values are of type string. `omegaframe` is determined by other parameters.
    """
    fargo_setups_file = pathlib.Path(fargo_setups_file)
    with fargo_setups_file.open("r") as f:
        fargo_setups = yaml.safe_load(f)
    fargo_setups = {k.lower(): v for k, v in fargo_setups.items()}

    # planet_config
    if "planetconfig" in fargo_setups:
        cfg_file = fargo_setups["planetconfig"].split("/")[1]
        planet_config = onet_disk2D.physics.read_planet_config(cfg_file)

        # update omegaframe, see fargo3d's docs for more explanations.
        fargo_setups["omegaframe"] = onet_disk2D.physics.get_frame_angular_velocity(
            frame=fargo_setups["frame"],
            omegaframe=float(fargo_setups["omegaframe"]),
            planet_distance=planet_config["Distance"],
        )
    else:
        planet_config = {}
        if fargo_setups["frame"] in ["G", "C"]:
            raise ValueError("No planet file, but omegaframe depends on a planet.")

    return fargo_setups, planet_config


def load_arg_groups(arg_groups_file):
    with pathlib.Path(arg_groups_file).open("r") as f:
        return yaml.safe_load(f)


def get_u_net_input_transform(
    col_idx_to_log: chex.Array, u_min: chex.Array, u_max: chex.Array
):
    """Get u_net_input_transform.

    Args:
        col_idx_to_log: bool array, whether to log transform the corresponding column.
        u_min: float array, minimum of u. The values are either in linear scale or log10 scale, following `col_idx_to_log`.
        u_max: float array, maximum of u. The values are either in linear scale or log10 scale, following `col_idx_to_log`.

    Returns:
        u_net_input_transform: jax function, transform u to u_net_input.
    """
    normalization_func = onet_disk2D.model.get_input_normalization(
        u_min=jnp.array(u_min), u_max=jnp.array(u_max)
    )

    @jax.jit
    def u_net_input_transform(inputs):
        inputs = onet_disk2D.utils.to_log(inputs, col_idx_to_log)
        inputs = normalization_func(inputs)
        return inputs

    return u_net_input_transform


class JOB:
    def __init__(self, args):
        # load args, configs
        self.args = vars(args) if isinstance(args, argparse.Namespace) else args
        self.fargo_setups, self.planet_config = load_fargo_setups(
            self.args["fargo_setups"]
        )
        # load arg_groups.yml
        # arg_groups is used for build bcs
        self.arg_groups = load_arg_groups(self.args["arg_groups_file"])

        # build model
        self.col_idx_to_log = jnp.array(
            [s == "log10" for s in self.args["u_transform"]]
        )
        self.s_raw_and_a_fn = onet_disk2D.model.outputs_scaling_transform(
            self.model.forward_apply
        )[1]
        self.state = {"scaling_factors": jnp.array(self.args["scale_on_s"])}

    @functools.cached_property
    def model(self):
        if self.args["mlp_layer_size"]:
            # use MLP rather than DeepONet
            if (
                self.args["u_net_layer_size"]
                or self.args["y_net_layer_size"]
                or self.args["z_net_layer_size"]
            ):
                raise ValueError
            else:
                m = onet_disk2D.model.build_mlponet(
                    layer_size=self.args["mlp_layer_size"],
                    Nx=len(self.parameter),
                    u_net_input_transform=self.u_net_input_transform,
                    y_net_input_transform=self.y_net_input_transform,
                    **self.args,
                )
        else:
            # DeepONet
            m = onet_disk2D.model.build_model(
                Nx=len(self.parameter),
                u_net_input_transform=self.u_net_input_transform,
                u_net_output_transform=self.u_net_output_transform,
                y_net_input_transform=self.y_net_input_transform,
                y_net_output_transform=self.y_net_output_transform,
                **self.args,
            )
        return m

    @functools.cached_property
    def ic(self):
        """

        Caution:
            When unknown == 'log_sigma', ic should not be used for output transformation or loss weighting. Please use the condition unknown == 'log_sigma' to avoid these situations.

        Returns:

        """
        parameter = self.fargo_setups.copy()
        # parameter input index
        for i, k in enumerate(self.parameter):
            if k in parameter:
                raise KeyError(f"{k} is already in parameter")
            else:
                parameter[k.lower()] = (i,)

        if self.args["unknown"] in ["log_sigma", "sigma"]:
            ic_ = onet_disk2D.physics.initial_condition.get_sigma_ic(
                self.fargo_setups["densityinitial"], parameter
            )
        elif self.args["unknown"] == "v_r":
            ic_ = onet_disk2D.physics.initial_condition.get_v_r_ic(
                self.fargo_setups["vyinitial"], parameter
            )
        elif self.args["unknown"] == "v_theta":
            ic_ = onet_disk2D.physics.initial_condition.get_v_theta_ic(
                self.fargo_setups["vxinitial"], parameter
            )
        else:
            raise NotImplementedError

        return ic_

    @functools.cached_property
    def unknown_type(self):
        if self.args["unknown"] == "log_sigma":
            return "sigma"
        else:
            return self.args["unknown"]

    @functools.cached_property
    def save_dir(self):
        save_dir = pathlib.Path(self.args["save_dir"])
        if not save_dir.exists():
            save_dir.mkdir()
        return save_dir

    @functools.cached_property
    def summary_dir(self):
        sum_dir = self.save_dir / "summary"
        if not sum_dir.exists():
            sum_dir.mkdir()
        return sum_dir

    @functools.cached_property
    def parameter(self):
        """Names of the parametric inputs to DeepONet.

        The names should be specified from command line to be compatible with pure physics training (no data).
        If the file `fargo_runs.yml` is available, the names here should be the same as the `parameters` values.
        If the data files are available, the names here should be the same as the names of non-dimensional coordinates.

        Returns:

        """
        ps = sorted(self.args["parameter"])
        if ps is None:
            raise ValueError
        return ps

    @functools.cached_property
    def u_net_input_transform(self):
        if len(self.args["u_min"]) != len(self.parameter):
            print("=" * 20)
            print(
                f"Warning: u_min = {self.args['u_min']}, parameter: {self.parameter}. Shapes does not match."
            )
            print("=" * 20)
        if len(self.args["u_max"]) != len(self.parameter):
            print("=" * 20)
            print(
                f"Warning: u_max = {self.args['u_max']}, parameter: {self.parameter}. Shapes does not match."
            )
            print("=" * 20)
        if len(self.args["u_transform"]) != len(self.parameter):
            raise ValueError

        return get_u_net_input_transform(
            self.col_idx_to_log, self.args["u_min"], self.args["u_max"]
        )

    @functools.cached_property
    def u_net_output_transform(self):
        return None

    @functools.cached_property
    def y_net_input_transform(self):
        return onet_disk2D.model.get_period_transform(
            r_min=float(self.fargo_setups["ymin"]),
            r_max=float(self.fargo_setups["ymax"]),
        )

    @functools.cached_property
    def y_net_output_transform(self):
        return None

    @functools.cached_property
    def s_pred_fn(self):
        """

        Returns:
            Callable: (params, scaling_factors, inputs)
        """
        s_fn = onet_disk2D.model.outputs_scaling_transform(self.model.forward_apply)[0]
        # Callable: (params, state, inputs)

        if self.args["ic_shift"] == "ON":
            if self.ic is None:
                raise ValueError(f"self.ic is None but ic_shift is ON.")
            if self.args["unknown"] == "log_sigma":
                raise NotImplementedError
            s_fn = onet_disk2D.physics.initial_condition.get_transformed_s_fn(
                self.ic, s_fn
            )
        elif self.args["ic_shift"] == "OFF":
            pass
        else:
            raise NotImplementedError

        return s_fn

    def load_model(self, model_dir):
        # load model from files and overwrite current model
        print(f"Loading trained model from {model_dir}")
        self.model.params = onet_disk2D.model.load_params(model_dir)
        self.state = onet_disk2D.model.load_state(model_dir)

    def predict(
        self,
        parameters,
        save_dir,
        ymin=None,
        ymax=None,
        ny=None,
        nx=None,
        name="",
        **kwargs,
    ):
        """

        Args:
            parameters: A dict of physics parameters
                The keywords are in uppercase
                values: shape (Nu, 1)
            save_dir:
            ymin:
            ymax:
            ny:
            nx:
            name:
            **kwargs:

        Returns:

        """

        ymin = ymin if ymin else float(self.fargo_setups["ymin"])
        ymax = ymax if ymax else float(self.fargo_setups["ymax"])
        ny = ny if ny else int(self.fargo_setups["ny"])
        nx = nx if nx else int(self.fargo_setups["nx"])
        # generate coords
        grids = onet_disk2D.grids.Grids(
            ymin=ymin, ymax=ymax, xmin=-np.pi, xmax=np.pi, ny=ny, nx=nx
        )

        u = jnp.concatenate(
            [parameters[pname] for pname in sorted(parameters)], axis=-1
        )

        inputs = {
            "u_net": u,
            "y_net": grids.coords_fargo_all[self.unknown_type].reshape((-1, 2)),
        }

        # join parameters and fixed_parameters
        if {k.lower() for k in parameters} != {k.lower() for k in self.parameter}:
            raise ValueError("Input parameters do not match self.parameter")

        attrs = self.args.copy()
        attrs["parameter"] = ",".join(attrs["parameter"])

        predict = self.s_pred_fn(self.model.params, self.state, inputs)
        predict = predict.reshape(u.shape[0], ny, nx)
        coords = {
            pname: ("run", pvalue.reshape(len(u)))
            for pname, pvalue in parameters.items()
        }
        coords.update(
            {
                "r": ("r", grids.r_fargo_all[self.unknown_type]),
                "theta": ("theta", grids.theta_fargo_all[self.unknown_type]),
            }
        )
        predict = xr.DataArray(
            predict,
            coords=coords,
            dims=["run", "r", "theta"],
            attrs=attrs,
        )
        if name:
            file_name = f"batch_predict_{self.args['unknown']}_{name}.nc"
        else:
            file_name = f"batch_predict_{self.args['unknown']}.nc"
        predict.to_netcdf(
            save_dir / file_name,
            format="NETCDF4",
            engine="netcdf4",
        )

    def test(
        self,
        data,
        data_type: str,
        save_dir,
    ):
        """

        Notes:
            The current implementation is memory inefficient. To improve, we can follow the steps 1). split the data into batches 2). calculate the errors 3). summarize the errors and save to files.

        Args:
            data: One of {sigma, v_r, v_theta}
            data_type: 'train', 'val', 'test' or 'train_and_val'
            save_dir:

        Returns:

        """
        data = data[self.args["unknown"]]

        summary_dir = save_dir / "summary"
        if not summary_dir.exists():
            summary_dir.mkdir()

        errors = {"l2": {}, "mse": {}, "norm": {}}

        n_run = len(data["run"])

        datadict = onet_disk2D.data.to_datadict(data)
        # datadict: {inputs: {u_net, y_net}, s}

        predict = []
        for i in tqdm.trange(n_run):
            inputs = {
                "u_net": datadict["inputs"]["u_net"][i : i + 1],
                "y_net": datadict["inputs"]["y_net"],
            }
            predict.append(self.s_pred_fn(self.model.params, self.state, inputs))
        predict = jnp.concatenate(predict)

        dims = ("run", "r", "theta")

        # ============================
        # normal scale error
        # ============================

        # normalized error
        # ic
        ic: onet_disk2D.physics.initial_condition.IC
        ic_values = self.ic.func(
            datadict["inputs"]["u_net"], datadict["inputs"]["y_net"]
        )

        # (pred - truth) / (truth - ic)
        if self.args["unknown"] == "log_sigma":
            # the data is already logged
            truth_normal_scale = 10.0 ** datadict["s"]
            predict_normal_scale = 10.0**predict
        else:
            truth_normal_scale = datadict["s"]
            predict_normal_scale = predict

        normalized_error = calculate_normalized_error(
            truth=truth_normal_scale, predict=predict_normal_scale, ic_values=ic_values
        )

        ms_normalized_errors = jnp.nanmean(normalized_error**2)
        # guild scalar
        print(f"{data_type}_{self.unknown_type}_norm: {ms_normalized_errors}")
        print(f"{data_type}_{self.unknown_type}_norm= {ms_normalized_errors:.2g}")
        errors["norm"][self.unknown_type] = f"{ms_normalized_errors:.2g}"

        l2_errors = jnp.linalg.norm(
            predict_normal_scale - truth_normal_scale
        ) / jnp.linalg.norm(truth_normal_scale)
        # guild scalar
        print(f"{data_type}_{self.unknown_type}_l2: {l2_errors}")
        print(f"{data_type}_{self.unknown_type}_l2= {l2_errors:.2g}")
        errors["l2"][self.unknown_type] = f"{l2_errors:.2g}"

        # mse for v_r and v_theta
        if self.unknown_type in ["v_r", "v_theta"]:
            mse_errors = jnp.nanmean((predict_normal_scale - truth_normal_scale) ** 2)
            # guild scalar
            print(f"{data_type}_{self.unknown_type}_mse: {mse_errors}")
            print(f"{data_type}_{self.unknown_type}_mse= {mse_errors:.2g}")
            errors["mse"][self.unknown_type] = f"{mse_errors:.2g}"

        to_file(
            truth=truth_normal_scale,
            predict=predict_normal_scale,
            error=normalized_error,
            shape=data.shape,
            coords=data.coords,
            data_type=data_type,
            unknown=self.unknown_type,
            save_dir=save_dir,
            dims=dims,
        )

        # ============================
        # log scale error
        # ============================
        if self.args["unknown"] in ["sigma", "log_sigma"]:
            k = "log_sigma"
            if self.args["unknown"] == "sigma":
                truth_log_scale = np.log10(datadict["s"])
                predict_log_scale = np.log10(predict)
            else:
                truth_log_scale = datadict["s"]
                predict_log_scale = predict

            error = predict_log_scale - truth_log_scale
            mean_squared_errors = jnp.nanmean(error**2)
            # guild scalar
            print(f"{data_type}_{k}_mse: {mean_squared_errors}")
            print(f"{data_type}_{k}_mse= {mean_squared_errors:.2g}")
            errors["mse"][k] = f"{mean_squared_errors:.2g}"

            to_file(
                truth=truth_log_scale,
                predict=predict_log_scale,
                error=error,
                shape=data.shape,
                coords=data.coords,
                data_type=data_type,
                unknown=k,
                save_dir=save_dir,
                dims=dims,
            )

        with open(summary_dir / f"{data_type}_error.yml", "w") as f:
            yaml.safe_dump(errors, f)


class Train(JOB):
    def __init__(self, args):
        super(Train, self).__init__(args)
        # save args
        self.save_args()
        # train related
        self.opt_state, self.opt_update = self.get_optimizer()
        self.loss_weights = {}
        self.vs = {}
        self.gs = {}

    def save_args(self):
        with (self.save_dir / "args.yml").open("w") as f:
            yaml.safe_dump(self.args, f)

    def get_optimizer(self):
        """Set optimzier for training.

        Notes:
            Please initialize models first as model.params is required here.

        """
        # learning rate
        transition_steps_on = self.args["transition_steps"] > 0
        decay_rate_on = self.args["decay_rate"] < 1.0
        learning_rate = self.args["lr"]
        if (not transition_steps_on) and (not decay_rate_on):
            # no exponential decay of learning rate
            pass
        elif transition_steps_on and decay_rate_on:
            learning_rate = optax.exponential_decay(
                learning_rate,
                transition_steps=self.args["transition_steps"],
                decay_rate=self.args["decay_rate"],
            )
        else:
            raise ValueError(
                "Missing arguments for exponential decay of learning rate."
            )
        # optimizer
        if self.args["optimizer"] == "adam":
            optimizer = optax.adam(learning_rate)
        else:
            raise NotImplementedError(f"optimizer = {self.args['optimizer']}")

        return optimizer.init(self.model.params), jax.jit(optimizer.update)

    @functools.cached_property
    def constraints(self):
        return None

    @functools.cached_property
    def callbacklist(self) -> onet_disk2D.callbacks.CallbackList:
        callbacks = []
        callbacks = onet_disk2D.callbacks.CallbackList(callbacks)
        callbacks.set_job(self)
        return callbacks

    def compute_total_g(self):
        if self.args["g_compute_method"] == "sum":
            return onet_disk2D.gradients.sum_gradients(list(self.gs.values()))
        elif self.args["g_compute_method"] in [
            "ntk_weighted_sum",
            "initial_loss_weighted_sum",
        ]:
            return onet_disk2D.gradients.sum_weighted_gradients(
                self.gs, self.loss_weights
            )
        else:
            raise NotImplementedError(
                f"g_compute_method = {self.args['g_compute_method']}"
            )

    def train(self):
        self.callbacklist.on_train_begin()
        for i_steps in tqdm.trange(self.args["steps"]):
            self.callbacklist.on_train_batch_begin(i_steps, i_steps)
            self.vs, self.gs = self.constraints.get_v_g(self.model.params, self.state)
            total_g = self.compute_total_g()
            updates, self.opt_state = self.opt_update(
                total_g, self.opt_state, self.model.params
            )
            self.model.params = optax.apply_updates(self.model.params, updates)
            self.callbacklist.on_train_batch_end(i_steps, i_steps)

        self.callbacklist.on_train_end()


def load_job_args(run_dir, args_file, arg_groups_file, fargo_setup_file):
    """Load args for restarting JOB from run_dir/args_file.

    And update the path of arg_groups_file and fargo_setup_file.
    """
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / args_file, "r") as f:
        job_args = yaml.safe_load(f)
    job_args["arg_groups_file"] = (run_dir / arg_groups_file).as_posix()
    job_args["fargo_setups"] = (run_dir / fargo_setup_file).as_posix()

    return job_args


def outliers_to_nan(array, a=10):
    """

    Args:
        array: shape: (Nu, Ny)
        a:

    Returns:

    """
    array = np.copy(array)
    q1 = np.nanpercentile(array, q=25, axis=-1, keepdims=True)
    q3 = np.nanpercentile(array, q=75, axis=-1, keepdims=True)
    iqr = q3 - q1
    upper = q3 + a * iqr
    lower = q1 - a * iqr
    # print(upper, lower)
    array[array < lower] = np.nan
    array[array > upper] = np.nan
    return array, upper, lower


def calculate_normalized_error(truth, predict, ic_values):
    """

    Args:
        truth: shape (Nu, Ny)
        predict: shape (Nu, Ny)
        ic_values:

    Returns:

    """
    normalized_error = (predict - truth) / (truth - ic_values)
    normalized_error = normalized_error.at[jnp.isinf(normalized_error)].set(jnp.nan)
    # remove outliers
    normalized_error, _, _ = outliers_to_nan(normalized_error, a=10)

    return normalized_error


def mean_coordinate_error_to_file(error, data_type, unknown, save_dir):
    error = error.mean(["r", "theta"])
    error.to_netcdf(
        save_dir / f"{data_type}_mean_errors_{unknown}.nc",
        format="NETCDF4",
        engine="netcdf4",
    )


def truth_pred_error_to_file(
    truth,
    predict,
    error,
    coords,
    data_type,
    unknown,
    save_dir,
    dims=("run", "r", "theta"),
):
    """

    Args:
        truth: array of shape (Nu, Nr, Ntheta)
        predict: array of shape (Nu, Nr, Ntheta)
        error: array of shape (Nu, Nr, Ntheta)
        dims:

    Returns:
        None
    """

    truth_pred_error = xr.Dataset(
        data_vars={
            "truth": (dims, truth),
            "pred": (dims, predict),
            "errors": (dims, error),
        },
        coords=coords,
    )

    file_path = save_dir / f"{data_type}_batch_truth_pred_{unknown}.nc"
    print(f"Saved to {file_path}")
    truth_pred_error.to_netcdf(
        file_path,
        format="NETCDF4",
        engine="netcdf4",
        encoding={
            "truth": {"dtype": "float32"},
            "pred": {"dtype": "float32"},
            "errors": {"dtype": "float32"},
        },
    )


def to_file(
    truth,
    predict,
    error,
    shape,
    coords,
    data_type: str,
    unknown: str,
    save_dir,
    dims=("run", "r", "theta"),
):
    """

    Args:
        truth: shape (Nu, Ny)
        predict: shape (Nu, Ny)
        error: shape (Nu, Ny)
        shape:
        coords:
        data_type:
        unknown:
        save_dir:
        dims:

    Returns:

    """
    truth = truth.reshape(shape)
    predict = predict.reshape(shape)
    error = error.reshape(shape)

    squared_error = xr.DataArray(error**2, coords=coords, dims=dims)

    mean_coordinate_error_to_file(
        error=squared_error,
        data_type=data_type,
        unknown=unknown,
        save_dir=save_dir,
    )

    truth_pred_error_to_file(
        truth=truth,
        predict=predict,
        error=error,
        coords=coords,
        data_type=data_type,
        unknown=unknown,
        save_dir=save_dir,
        dims=dims,
    )
