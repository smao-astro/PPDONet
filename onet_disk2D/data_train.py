"""
Script for training
"""

import onet_disk2D.run
import onet_disk2D.train


def get_parser():
    parser = onet_disk2D.train.get_parser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory from which to load batch_truth_sigma.nc, batch_truth_v_theta.nc and batch_truth_v_r.nc",
    )
    # train
    parser.add_argument(
        "--train_sample_percent",
        type=float,
        default=0.9,
        help="Percentage of training samples in the whole dataset. (1-`train_sample_percent`) is the percentage of validation samples.",
    )
    parser.add_argument("--batch_size_data", type=int, default=4)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    job = onet_disk2D.run.DataTrain(args)
    job.train()
    job.test(job.train_data, data_type="train", model_dir=job.save_dir)
    job.test(job.val_data, data_type="val", model_dir=job.save_dir)
