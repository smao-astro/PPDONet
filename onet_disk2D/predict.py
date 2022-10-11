import argparse
import pathlib

import pandas as pd
import yaml

import onet_disk2D.run
import onet_disk2D.train


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
        help="Directory that store model files (params_struct.pkl, params.npy, etc).",
    )
    # parameter
    parser.add_argument(
        "--parameter_file",
        type=str,
        help="pandas DataFrame file (csv) that keeps parameters to predict.",
    )

    parser.add_argument(
        "--save_dir", type=str, default="", help="If empty, save to train directory."
    )
    # inputs
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--ny", type=int, help="NY in fargo3d.")
    parser.add_argument("--nx", type=int, help="NX in fargo3d.")
    parser.add_argument(
        "--name", type=str, default="", help="Name attribute for output file."
    )

    return parser


def get_parameter_values(parameter_file):
    parameters = pd.read_csv(parameter_file, index_col=0)
    parameters = {k: series.values[..., None] for k, series in parameters.items()}
    return parameters


if __name__ == "__main__":
    predict_args = get_parser().parse_args()
    run_dir = pathlib.Path(predict_args.run_dir).resolve()
    if predict_args.model_dir:
        model_dir = pathlib.Path(predict_args.model_dir).resolve()
    else:
        model_dir = run_dir

    # load args from file
    with open(predict_args.args_file, "r") as f:
        train_args = yaml.safe_load(f)
    train_args["arg_groups_file"] = (run_dir / predict_args.arg_groups_file).as_posix()
    train_args["fargo_setups"] = (run_dir / predict_args.fargo_setup_file).as_posix()

    job = onet_disk2D.run.JOB(train_args)

    # parameters
    parameter_values = get_parameter_values(predict_args.parameter_file)

    job.predict(
        parameters=parameter_values,
        model_dir=model_dir,
        save_dir=predict_args.save_dir,
        ymin=predict_args.ymin,
        ymax=predict_args.ymax,
        ny=predict_args.ny,
        nx=predict_args.nx,
        name=predict_args.name,
    )
