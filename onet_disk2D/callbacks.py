"""Flexible manipulating job attributes in a clean way."""
import typing

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

import onet_disk2D.data
import onet_disk2D.model


class Callback:
    def __init__(self):
        self.job = None

    def set_job(self, job):
        self.job = job

    # Global methods
    def on_train_begin(self):
        pass

    def on_test_begin(self):
        pass

    def on_predict_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_test_end(self):
        pass

    def on_predict_end(self):
        pass

    # Batch-level methods
    def on_train_batch_begin(self, i_steps, i_steps_total):
        pass

    def on_test_batch_begin(self, i_steps, i_steps_total):
        pass

    def on_predict_batch_begin(self, i_steps, i_steps_total):
        pass

    def on_train_batch_end(self, i_steps, i_steps_total):
        pass

    def on_test_batch_end(self, i_steps, i_steps_total):
        pass

    def on_predict_batch_end(self, i_steps, i_steps_total):
        pass

    # Epoch-level methods
    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass


class CallbackList:
    def __init__(self, callbacks: typing.List[Callback]):
        self.callbacks: typing.List[Callback] = callbacks
        self.job = None

    def set_job(self, job):
        self.job = job
        for callback in self.callbacks:
            callback.set_job(job)

    # Global methods
    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_test_begin(self):
        for callback in self.callbacks:
            callback.on_test_begin()

    def on_predict_begin(self):
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_test_end(self):
        for callback in self.callbacks:
            callback.on_test_end()

    def on_predict_end(self):
        for callback in self.callbacks:
            callback.on_predict_end()

    # Batch-level methods
    def on_train_batch_begin(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_train_batch_begin(i_steps, i_steps_total)

    def on_test_batch_begin(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_test_batch_begin(i_steps, i_steps_total)

    def on_predict_batch_begin(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_predict_batch_begin(i_steps, i_steps_total)

    def on_train_batch_end(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_train_batch_end(i_steps, i_steps_total)

    def on_test_batch_end(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_test_batch_end(i_steps, i_steps_total)

    def on_predict_batch_end(self, i_steps, i_steps_total):
        for callback in self.callbacks:
            callback.on_predict_batch_end(i_steps, i_steps_total)

    # Epoch-level methods
    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)


class Sampler(Callback):
    def __init__(self, steps_per_resample, key=jax.random.PRNGKey(123)):
        super(Sampler, self).__init__()
        self.steps_per_resample = steps_per_resample
        self.key = key

    def on_train_batch_begin(self, i_steps, i_steps_total):
        if i_steps_total % self.steps_per_resample == 0:
            _, self.key = jax.random.split(self.key, 2)
            self.job.constraints.resample(self.key)


class InverseInitialLossWeighting(Callback):
    def __init__(self, file_name):
        super(InverseInitialLossWeighting, self).__init__()
        self.file_path = file_name

    def set_job(self, job):
        super(InverseInitialLossWeighting, self).set_job(job)
        self.file_path = self.job.summary_dir / self.file_path

    def on_train_begin(self):
        self.job.loss_weights = {k: 1.0 for k in self.job.constraints.loss_fn}

    def on_train_batch_end(self, i_steps, i_steps_total):
        if i_steps_total == 0:
            self.job.loss_weights = jax.tree_map(lambda x: 1.0 / x, self.job.vs)
            # log
            loss_weights = jax.tree_map(float, self.job.loss_weights)
            with self.file_path.open("w") as f:
                yaml.safe_dump(loss_weights, f)


def plot_loss(loss_xarray) -> plt.Figure:
    ax: plt.Axes
    fig, ax = plt.subplots()
    for key, value in loss_xarray.items():
        ax.plot(loss_xarray.i_steps, value, label=key)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    plt.tight_layout()
    return fig


def plot_multi_group_loss(
    loss_xarray,
    candidate_group=("data",),
    figsize=plt.rcParamsDefault["figure.figsize"],
) -> plt.Figure:
    group = []
    for gkey in candidate_group:
        key: str
        for key in loss_xarray:
            if key.startswith(gkey):
                group.append(gkey)
                break
    group = sorted(group)

    ax: plt.Axes
    fig: plt.Figure
    fig, axes = plt.subplots(
        ncols=len(group),
        sharey="row",
        figsize=(figsize[0] * len(group) * 0.5, figsize[1] * 0.6),
        constrained_layout=True,
    )
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for gkey, ax in zip(group, axes):
        for key, value in loss_xarray.items():
            if key.startswith(gkey):
                ax.plot(loss_xarray.i_steps, value, label=key)
        ax.legend()
        ax.set_yscale("log")
    fig.supxlabel("Steps")
    fig.supylabel("MSE Loss")

    return fig


def plot_mag(mag_xarray) -> plt.Figure:
    ax: plt.Axes
    fig, ax = plt.subplots()
    for key, value in mag_xarray.items():
        ax.plot(mag_xarray.i_steps, value, label=key)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("steps")
    ax.set_ylabel("Raw output mag")
    return fig


class LossLogger(Callback):
    def __init__(
        self, file_name, train_data_loader, val_data_loader, period=10, period_dump=300
    ):
        super(LossLogger, self).__init__()
        self.file_name = file_name
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.period = period
        self.period_dump = period_dump
        self.log = {}

    def dump(self):
        # to guild scalar
        print(f'step: {self.log["i_steps"][-1]}')
        for k, v in self.log.items():
            if k == "i_steps":
                continue
            print(f"{k}_loss: {v[-1]:.2g}")

        # to xarray
        coords = {"i_steps": self.log["i_steps"]}
        loss = {k: (["i_steps"], v) for k, v in self.log.items() if k != "i_steps"}
        loss_xarray = xr.Dataset(data_vars=loss, coords=coords)
        loss_xarray.to_netcdf(self.job.save_dir / (self.file_name + ".nc"))

    def on_train_begin(self):
        self.log.update({"train_" + k: [] for k in self.job.constraints.loss_fn})
        self.log.update({"val_" + k: [] for k in self.job.constraints.loss_fn})
        self.log["i_steps"] = []

    def on_train_batch_end(self, i_steps, i_steps_total):
        if i_steps_total % self.period == 0:
            # check
            if jnp.any(jnp.isnan(jnp.stack(list(self.job.vs.values())))):
                print("loss: ", self.job.vs)
                raise ValueError("Get NAN in training!")
            self.log["i_steps"].append(i_steps_total)
            for k, loss_fn in self.job.constraints.loss_fn.items():
                train_loss = []
                for i, data in zip(
                    range(self.train_data_loader.n_batch), self.train_data_loader
                ):
                    train_loss.append(
                        loss_fn(
                            self.job.model.params,
                            self.job.state,
                            data[k[5:]],
                        )
                    )
                self.log["train_" + k].append(float(np.mean(train_loss)))
                val_loss = []
                for i, data in zip(
                    range(self.val_data_loader.n_batch), self.val_data_loader
                ):
                    val_loss.append(
                        loss_fn(
                            self.job.model.params,
                            self.job.state,
                            data[k[5:]],
                        )
                    )
                self.log["val_" + k].append(float(np.mean(val_loss)))
        if i_steps_total % self.period_dump == 0:
            self.dump()

    def on_train_end(self):
        self.dump()


class RawOutputMagLogger(Callback):
    def __init__(self, unknown, file_name, period=10):
        super(RawOutputMagLogger, self).__init__()
        self.unknown = unknown
        self.period = period
        self.file_name = file_name
        self.log = {"i_steps": [], "mag": []}

    def on_train_batch_end(self, i_steps, i_steps_total):
        if i_steps_total % self.period == 0:
            samples = self.job.constraints.samples[f"data_{self.unknown}"]["inputs"]

            self.log["i_steps"].append(i_steps_total)
            _, mag = self.job.s_raw_and_a_fn(
                self.job.model.params,
                self.job.state,
                samples,
            )
            self.log["mag"].append(mag)

    def on_train_end(self):
        coords = {"i_steps": self.log["i_steps"]}
        mag = jnp.stack(self.log["mag"])
        mag = {self.unknown: (["i_steps"], mag)}
        mag = xr.Dataset(data_vars=mag, coords=coords)
        mag.to_netcdf(self.job.save_dir / (self.file_name + ".nc"))

        fig = plot_mag(mag)
        fig.savefig(self.job.summary_dir / (self.file_name + ".png"), format="png")


class ModelSaver(Callback):
    def __init__(self, period=100):
        super(ModelSaver, self).__init__()
        self.period = period

    def save_model(self, name):
        if name != "final":
            save_dir = self.job.save_dir / name
            if not save_dir.exists():
                save_dir.mkdir()
        else:
            save_dir = self.job.save_dir
        onet_disk2D.model.save_params(self.job.model.params, save_dir=save_dir)
        onet_disk2D.model.save_state(self.job.state, save_dir=save_dir)

    def on_train_batch_end(self, i_steps, i_steps_total):
        if i_steps_total % self.period == 0:
            self.save_model(name=f"steps_{i_steps_total}")

    def on_train_end(self):
        self.save_model(name="final")


class InputChecker(Callback):
    def on_train_begin(self):
        key = self.job.args["unknown"]
        data = onet_disk2D.data.to_datadict(self.job.data[key])
        transformed_inputs = self.job.u_net_input_transform(data["inputs"]["u_net"])
        print("transformed_inputs:")
        print("\tMean: ", jnp.mean(transformed_inputs, axis=0))
        print("\tStd: ", jnp.std(transformed_inputs, axis=0))
