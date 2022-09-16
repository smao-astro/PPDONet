"""
Script for training
"""
import argparse


def list_of_float(raw_inputs: str):
    return [float(value) for value in raw_inputs.split(",")]


def list_of_int(raw_inputs: str):
    return [int(value) for value in raw_inputs.split(",")]


def list_of_str(raw_inputs: str):
    return [value for value in raw_inputs.split(",")]


def get_parser():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("--fargo_setups", type=str, default="fargo_setups.yml")
    parser.add_argument("--arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument("--save_dir", type=str, default=".")

    # model
    parser.add_argument("--parameter", type=list_of_str)
    parser.add_argument(
        "--unknown",
        default="log_sigma",
        choices=["sigma", "log_sigma", "v_r", "v_theta"],
    )
    parser.add_argument(
        "--Nnode",
        type=int,
        default=5,
        help="Number of neurons of the last layer of u_net (and y_net).",
    )
    parser.add_argument("--u_net_layer_size", type=list_of_int, default="10,20")
    parser.add_argument("--y_net_layer_size", type=list_of_int, default="15,25")
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["sin", "tanh", "swish", "stan"],
    )
    parser.add_argument(
        "--initializer",
        type=str,
        choices=[
            "glorot_uniform",
            "glorot_normal",
            "lecun_uniform",
            "lecun_normal",
            "he_uniform",
            "he_normal",
            "sine_uniform",
        ],
        default="glorot_normal",
    )
    # periodic boundary hard constraint defaults to ON, no need to add to args
    parser.add_argument("--u_min", type=list_of_float, default="1.0")
    parser.add_argument("--u_max", type=list_of_float, default="1.0")
    parser.add_argument("--u_transform", type=list_of_str, default="")
    parser.add_argument("--scale_on_s", type=float, default=1.0)
    # IC shift
    parser.add_argument(
        "--ic_shift",
        type=str,
        default="ON",
        choices=["ON", "OFF"],
        help="Shift the final outputs of DeepONet based on priors of initial conditions.",
    )

    # train
    # optimizer
    parser.add_argument(
        "--key", type=int, default=9999, help="Key to generate random numbers."
    )
    parser.add_argument("--optimizer", type=str, choices=["adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transition_steps", type=int, default=0)
    parser.add_argument("--decay_rate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--steps_per_resample", type=int, default=4)
    parser.add_argument("--steps_per_log", type=int, default=10)
    parser.add_argument("--steps_per_dump_log", type=int, default=3000)
    parser.add_argument("--steps_per_save_model", type=int, default=50)
    parser.add_argument(
        "--steps_per_log_out_mag",
        type=int,
        default=10,
        help="Number of steps between every logging of raw-output magnitude. The values are monitored to make sure the scales of raw outputs are around ones.",
    )
    parser.add_argument(
        "--g_compute_method",
        type=str,
        default="sum",
        choices=["sum", "ntk_weighted_sum", "initial_loss_weighted_sum"],
    )
    return parser
