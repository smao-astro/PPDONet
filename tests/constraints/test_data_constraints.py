import pathlib

import jax.numpy as jnp
import jax.random
import jaxphyinf.model
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots
import pytest

import onet_disk2D.constraints
import onet_disk2D.data
import onet_disk2D.model
import onet_disk2D.physics.initial_condition
import onet_disk2D.vmap

SINGLEP_DATAPATH = "../data/singlep"
SINGLEP_FIXEDP = {
    "sigma0": 1.0,
    "nu": 1e-5,
    "sigmaslope": 0.0,
    "flaringindex": 0.0,
    "aspectratio": 0.05,
}
MULTIP_DATAPATH = "../data/multip"
MULTIP_FIXEDP = {
    "sigma0": 1.0,
    "nu": 1e-5,
    "sigmaslope": 0.0,
    "flaringindex": 0.0,
}


class TestDataConstraints:
    @pytest.fixture
    def u_net_layer_size(self):
        return [10, 20]

    @pytest.fixture
    def y_net_layer_size(self):
        return [15, 30]

    @pytest.fixture
    def n_node(self):
        return 4

    @pytest.fixture
    def u(self):
        """Parameter PLANETMASS"""
        return jnp.linspace(1e-4, 1e-6, 5)[:, None]

    @pytest.fixture
    def y(self):
        r_min = 0.4
        r_max = 2.5
        Nr = 5
        Ntheta = 3
        r = jnp.linspace(r_min, r_max, Nr)
        theta_min = -jnp.pi
        theta_max = jnp.pi
        theta = jnp.linspace(theta_min, theta_max, Ntheta)

        y = jnp.stack(jnp.meshgrid(r, theta, indexing="ij"), axis=-1)
        y = y.reshape((-1, 2))
        return y

    @pytest.fixture(params=["log_sigma", "sigma"])
    def unknown(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, u, y):
        return {"u_net": u, "y_net": y}

    @pytest.fixture(
        params=[(SINGLEP_DATAPATH, SINGLEP_FIXEDP), (MULTIP_DATAPATH, MULTIP_FIXEDP)]
    )
    def dataloader(self, unknown, request):
        datapath, fixed_parameters = request.param
        datapath = pathlib.Path(datapath)
        data = onet_disk2D.data.load_last_frame_data(
            datapath,
            unknown=unknown,
        )
        batch_size = 9
        return onet_disk2D.data.DataIterLoader(
            data, batch_size, fixed_parameters=fixed_parameters
        )

    @pytest.fixture
    def scaling_factors(self):
        return {"scaling_factors": jnp.array(0.1)}

    @pytest.fixture
    def model(self, n_node, y_net_layer_size, u_net_layer_size, dataloader):
        return onet_disk2D.model.build_model(
            Nnode=n_node,
            u_net_layer_size=u_net_layer_size,
            y_net_layer_size=y_net_layer_size,
            Nx=len(dataloader.parameter_names),
        )

    @pytest.fixture
    def ic(self):
        return onet_disk2D.physics.initial_condition.PowerlawSigmaIC()

    @pytest.fixture
    def s_fn(self, model, ic):
        f = jaxphyinf.model.outputs_scaling_transform(model.forward_apply)[0]
        f = onet_disk2D.physics.initial_condition.get_transformed_s_fn(ic, f)
        return f

    @pytest.fixture
    def data_constraints(self, s_fn, unknown, dataloader):
        return onet_disk2D.constraints.DataConstraints(
            s_fn, unknown=unknown, dataloader=dataloader
        )

    @pytest.fixture
    def weighted_data_constraints(self, s_fn, unknown, dataloader, ic):
        return onet_disk2D.constraints.DataConstraints(
            s_pred_fn=s_fn,
            unknown=unknown,
            dataloader=dataloader,
            ic=ic,
            data_loss_weighting="mag",
        )

    @pytest.fixture(params=["data_constraints", "weighted_data_constraints"])
    def constraints(self, request):
        return request.getfixturevalue(request.param)

    def test_inputs(self, inputs):
        u = inputs["u_net"]
        y = inputs["y_net"]
        cri = [u.shape == (5, 1), y.shape == (15, 2)]
        assert all(cri)

    def test_WeightedDataLoss(self, unknown, dataloader, s_fn, scaling_factors, model):
        if unknown == "log_sigma":
            assert True
        else:
            ic = onet_disk2D.physics.initial_condition.PowerlawSigmaIC()
            loss = onet_disk2D.constraints.data_constraints.WeightedDataLoss(
                s_fn=s_fn, ic_fn=ic.func, data_loss_weighting="diff2"
            )
            data = iter(dataloader)
            parameters, data = next(data)
            l = loss.loss_fn(model.params, scaling_factors, parameters, data[unknown])
            assert l > 0

    def test_WeightedDataLoss_2(
        self, unknown, dataloader, s_fn, scaling_factors, model
    ):
        if unknown == "log_sigma":
            assert True
        else:
            ic = onet_disk2D.physics.initial_condition.PowerlawSigmaIC()
            loss = onet_disk2D.constraints.data_constraints.WeightedDataLoss(
                s_fn=s_fn, ic_fn=ic.func, data_loss_weighting="diff2"
            )
            data = iter(dataloader)
            parameters, data = next(data)
            diff = loss.diff2_fn(parameters, data["sigma"])
            diff = diff.reshape((-1, 128, 384))
            for d in diff:
                plt.figure()
                plt.imshow(d, cmap="Reds")
                plt.colorbar()
                plt.show()
            assert True

    def test_WeightedDataLoss_mag(
        self, unknown, dataloader, s_fn, scaling_factors, model
    ):
        if unknown == "log_sigma":
            assert True
        else:
            ic = onet_disk2D.physics.initial_condition.PowerlawSigmaIC()
            loss = onet_disk2D.constraints.data_constraints.WeightedDataLoss(
                s_fn=s_fn, ic_fn=ic.func, data_loss_weighting="mag"
            )
            data = iter(dataloader)
            parameters, data = next(data)
            w = loss.w_fn(parameters, data["sigma"])
            res = loss.res_fn(model.params, scaling_factors, parameters, data["sigma"])
            res2 = jnp.mean(res**2, axis=-1)

            df = pd.DataFrame(
                {
                    "pm": parameters["planetmass"][:, 0],
                    "w": w[:, 0],
                    "res2": res2,
                    "loss": w[:, 0] * res2,
                }
            )
            df.sort_values(by="pm", inplace=True)
            fig = plotly.subplots.make_subplots(rows=1, cols=2)
            fig.add_trace(go.Scatter(x=df["pm"], y=df["w"], name="w"), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=df["pm"], y=df["res2"], name="res2"), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df["pm"], y=df["loss"], name="loss"), row=1, col=2
            )
            fig.update_yaxes(dict(showexponent="all", exponentformat="e", type="log"))
            fig.update_xaxes(dict(showexponent="all", exponentformat="e"))
            fig.show()
            assert True

    def test_loss_fn(self, model, scaling_factors, constraints):
        constraints.resample(123)
        loss = {
            k: loss_fn(
                model.params,
                scaling_factors,
                constraints.parameters[k],
                constraints.samples[k],
            )
            for k, loss_fn in constraints.loss_fn.items()
        }
        assert set(loss) == {"data_" + constraints.unknown}

    def test_res_fn(self, model, scaling_factors, constraints):
        key = jax.random.PRNGKey(999)

        constraints.resample(key)
        res = {
            k: fn(
                model.params,
                scaling_factors,
                constraints.parameters[k],
                constraints.samples[k],
            )
            for k, fn in constraints.res_fn.items()
        }
        cri = [v.shape == constraints.samples[k]["s"].shape for k, v in res.items()]

        assert all(cri)

    def test_res_fn_vmap(self, model, scaling_factors, constraints):
        key = jax.random.PRNGKey(999)

        constraints.resample(key)
        name = "data_" + constraints.unknown
        res_fn = constraints.res_fn[name]
        res_fn_adaptive_vmap = onet_disk2D.vmap.adaptive_vmap_p_s(res_fn)
        parameters = constraints.parameters[name]
        samples = constraints.samples[name]

        res = res_fn(model.params, scaling_factors, parameters, samples)

        res_adaptive_vmap = res_fn_adaptive_vmap(
            model.params, scaling_factors, parameters, samples
        )

        cri = [jnp.all(jnp.isclose(res, res_adaptive_vmap))]

        assert all(cri)

    def test_dres_fn(self, model, scaling_factors, constraints):
        key = jax.random.PRNGKey(999)

        constraints.resample(key)
        name = "data_" + constraints.unknown
        res_fn = constraints.res_fn[name]
        parameters = constraints.parameters[name]
        samples = constraints.samples[name]

        dres_fn = jax.jacfwd(res_fn, argnums=-1)
        dres_adaptive_vmap_fn = onet_disk2D.vmap.adaptive_vmap_p_s(dres_fn)

        dres_adaptive_vmap = dres_adaptive_vmap_fn(
            model.params, scaling_factors, parameters, samples
        )

        cri = [
            dres_adaptive_vmap["inputs"]["y_net"].shape
            == (
                samples["inputs"]["u_net"].shape[0],
                samples["inputs"]["y_net"].shape[0],
                2,
            ),
            dres_adaptive_vmap["inputs"]["u_net"].shape
            == (
                samples["inputs"]["u_net"].shape[0],
                samples["inputs"]["y_net"].shape[0],
                samples["inputs"]["u_net"].shape[-1],
            ),
            dres_adaptive_vmap["s"].shape
            == (
                samples["inputs"]["u_net"].shape[0],
                samples["inputs"]["y_net"].shape[0],
            ),
        ]

        assert all(cri)
