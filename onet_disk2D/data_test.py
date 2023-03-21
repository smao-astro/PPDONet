"""Test trained models on data."""
import argparse
import pathlib

import yaml

import onet_disk2D.data
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
        "--data_type",
        type=str,
        default="test",
        # choices=["train", "val", "test", "train_and_val"],
    )
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

    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        test_args.args_file,
        test_args.arg_groups_file,
        test_args.fargo_setup_file,
    )

    job = onet_disk2D.run.JOB(job_args)
    job.load_model(model_dir)

    save_dir = onet_disk2D.run.setup_save_dir(test_args.save_dir, model_dir)

    # load test data
    test_data = onet_disk2D.data.load_last_frame_data(
        data_dir=test_args.data_dir,
        unknown=job.args["unknown"],
        parameter=job.parameter,
    )

    job.test(
        data=test_data,
        data_type=test_args.data_type,
        save_dir=save_dir,
    )
