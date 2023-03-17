import functools

import numpy as np

import onet_disk2D.callbacks
import onet_disk2D.constraints
import onet_disk2D.data
from .job import Train
import sklearn.model_selection


class DataTrain(Train):
    def __init__(self, args):
        super(DataTrain, self).__init__(args)
        # train related
        self.data = onet_disk2D.data.load_last_frame_data(
            data_dir=self.args["data_dir"],
            unknown=self.args["unknown"],
            parameter=self.parameter,
        )
        self.train_data, self.val_data = self.get_train_val_data()

        if set(self.data_loader.parameter_names) != set(self.parameter):
            raise ValueError

    def release_data(self):
        for d in self.data.values():
            d.close()

    def get_train_val_data(self):
        """

        Returns:
            train_data:
                a dict of xr.DataArray for training
                    key: One of log_sigma, sigma, v_r, v_theta
            val_data:
                a dict of xr.DataArray for training
                    key: One of log_sigma, sigma, v_r, v_theta
        """
        run = self.data[self.args["unknown"]]["run"].values
        train_run, val_run = sklearn.model_selection.train_test_split(
            run,
            train_size=self.args["train_sample_percent"],
            random_state=self.args["key"],
            shuffle=True,
        )
        train_data = {k: v.sel(run=train_run) for k, v in self.data.items()}
        val_data = {k: v.sel(run=val_run) for k, v in self.data.items()}
        return train_data, val_data

    @functools.cached_property
    def data_loader(self):
        return onet_disk2D.data.DataIterLoader(
            data=self.train_data,
            batch_size=self.args["batch_size_data"],
            fixed_parameters=self.fixed_parameters,
        )

    @functools.cached_property
    def val_data_loader(self):
        n_run = len(self.val_data[self.args["unknown"]]["run"])
        return onet_disk2D.data.DataIterLoader(
            data=self.val_data,
            batch_size=n_run,
            fixed_parameters=self.fixed_parameters,
        )

    @functools.cached_property
    def constraints(self):
        return onet_disk2D.constraints.DataConstraints(
            s_pred_fn=self.s_pred_fn,
            unknown=self.args["unknown"],
            dataloader=self.data_loader,
            ic=self.ic,
            data_loss_weighting=self.args["data_loss_weighting"],
        )

    @functools.cached_property
    def callbacklist(self) -> onet_disk2D.callbacks.CallbackList:
        callbacks = [onet_disk2D.callbacks.Sampler(self.args["steps_per_resample"])]

        if self.args["g_compute_method"] == "initial_loss_weighted_sum":
            callbacks.append(
                onet_disk2D.callbacks.InverseInitialLossWeighting(
                    "data_loss_weights.yml"
                )
            )
        elif self.args["g_compute_method"] == "ntk_weighted_sum":
            raise NotImplementedError

        callbacks.append(
            onet_disk2D.callbacks.LossLogger(
                "data_loss",
                train_data_loader=self.data_loader,
                val_data_loader=self.val_data_loader,
                period=self.args["steps_per_log"],
                period_dump=self.args["steps_per_dump_log"],
            )
        )

        callbacks.append(
            onet_disk2D.callbacks.ModelSaver(period=self.args["steps_per_save_model"])
        )
        callbacks.append(
            onet_disk2D.callbacks.RawOutputMagLogger(
                unknown=self.args["unknown"],
                file_name="raw_output_mag",
                period=self.args["steps_per_log_out_mag"],
            )
        )

        callbacks.append(onet_disk2D.callbacks.InputChecker())

        callbacks = onet_disk2D.callbacks.CallbackList(callbacks)
        callbacks.set_job(self)
        return callbacks

    def train(self):
        super(DataTrain, self).train()
        self.release_data()
