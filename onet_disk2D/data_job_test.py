"""Test trained models on data.

Files requirements:

- run_dir
    - args.yml
    - fargo_setups.yml
    - arg_groups.yml

- model_dir
    - params.npy
    - params_struct.pkl
    - state.npy
    - state_struct.pkl
# if model_dir is not specified, model_dir = run_dir

- data_dir
    - batch_truth_sigma.nc
    - batch_truth_v_r.nc
    - batch_truth_v_theta.nc

- save_dir
# if save_dir is not specified, save_dir = model_dir

"""
import argparse
import pathlib

import yaml

import onet_disk2D.run


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
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory that store model files (params_struct.pkl, params.npy, etc).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="If empty, save to model_dir. See onet_disk2D.run.job.JOB.test for more details.",
    )

    return parser


if __name__ == "__main__":
    test_args = get_parser().parse_args()
    run_dir = pathlib.Path(test_args.run_dir).resolve()
    if test_args.model_dir:
        model_dir = pathlib.Path(test_args.model_dir).resolve()
    else:
        model_dir = run_dir

    # load args from file
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        test_args.args_file,
        test_args.arg_groups_file,
        test_args.fargo_setup_file,
    )
    # replace train_args["data_dir"] with local path
    job_args["data_dir"] = test_args.data_dir
    # Warning: do not use train_args['save_dir']

    job = onet_disk2D.run.DataTrain(job_args)
    job.load_model(model_dir)

    save_dir = onet_disk2D.run.setup_save_dir(test_args.save_dir, model_dir)


    job.test(
        data=job.train_data,
        data_type="train",
        save_dir=save_dir,
    )

    job.test(
        data=job.val_data,
        data_type="val",
        save_dir=save_dir,
    )
