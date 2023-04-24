"""
Script for training
"""

import onet_disk2D.run
import onet_disk2D.train


def get_parser():
    parser = onet_disk2D.train.get_parser()
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=".",
        help="Directory from which to load batch_truth_sigma.nc, batch_truth_v_theta.nc and batch_truth_v_r.nc",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=".",
        help="Directory from which to load batch_truth_sigma.nc, batch_truth_v_theta.nc and batch_truth_v_r.nc",
    )
    parser.add_argument("--batch_size_train", type=int, default=4)
    parser.add_argument("--batch_size_val", type=int, default=4)
    parser.add_argument(
        "--data_loss_weighting",
        type=str,
        default="",
        choices=["", "diff2", "mag"],
        help="Assigning weights for different data points (on grids)."
        # The choice `diff2` weights residuals by (s_data-s_ic)**2."
        # The choice `mag` balances the importance of fargo runs by the magnitudes of fargo features.
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    job = onet_disk2D.run.DataTrain(args)
    job.train()
    job.test(job.train_data, data_type="train", save_dir=job.save_dir)
    job.test(job.val_data, data_type="val", save_dir=job.save_dir)
