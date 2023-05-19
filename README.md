# PPDONet: Deep Operator Networks for Fast Prediction of Solutions in Disk-Planet Systems

We develop a tool, which we name Protoplanetary Disk Operator Network (PPDONet), that can predict the solution of disk-planet interactions in protoplanetary disks in real-time. We base our tool on Deep Operator Networks (DeepONets), a class of neural networks capable of learning non-linear operators to represent deterministic and stochastic differential equations.
With PPDONet we map three scalar parameters in a disk-planet system -- the Shakura & Sunyaev viscosity $\alpha$, the disk aspect ratio $h_0$, and the planet-star mass ratio $q$ -- to steady-state solutions of the disk surface density, radial velocity, and azimuthal velocity.  We demonstrate the accuracy of the PPDONet solutions using a comprehensive set of tests. Our tool is able to predict the outcome of disk-planet interaction for one system in less than a second on a laptop, speeding the calculation by many orders of magnitude compared with conventional numerical solvers. A public implementation of PPDONet is available at \url{https://github.com/smao-astro/PPDONet}.

## Web app demo

We have developed a web app demo for PPDONet. Please visit [this link](https://ppdonet-1.herokuapp.com) to try it out!

![](https://github.com/smao-astro/PPDONet/blob/master/webapp.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

[//]: # (See deployment for notes on how to deploy the project on a live system.)

### Prerequisites

```
Python 3.9
```

### Installing

First, clone the repository:

```
git clone https://github.com/smao-astro/PPDONet.git
```

Then, install the required packages:

First create a virtual environment, then activate it, and run the following command:

```
pip install -r requirements.txt
```

## Examples

### Web app demo

```
python real_time_prediction.py \
--sigma_run_dir trained_network/single_log_sigma \
--v_r_run_dir trained_network/single_v_r \
--v_theta_run_dir trained_network/single_v_theta \
--nxy 256
```
And open the link you get from the terminal in your browser.

### Predicting the solutions of a batch of disk-planet systems

```
python -m onet_disk2D.predict \
--run_dir trained_network/single_log_sigma \
--parameter_file parameter_examples.csv \
--save_dir trained_network/single_log_sigma/predictions \
--num_cell_radial 200 \
--num_cell_azimuthal 600 \
--name pred
```
Please have a look at the `parameter_examples.csv` file for the format of the input parameter file. Add more rows of parameters as you wish. Be aware that the predictions are warranted only for the parameter ranges that the network is trained on.

## Built With

## Contributing

## Versioning

## The team

This software was developed by [Shunyuan Mao](https://github.com/smao-astro) under the supervision of Prof. Ruobing Dong at University of Victoria from 2022 to 2023.

## License

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

## Acknowledgments

The following people have contributed to the project:

- **Lu Lu** - Department of Chemical and Biomolecular Engineering, University of Pennsylvania, Philadelphia, PA 19104, USA
- **Kwang Moo Yi** - Department of Computer Science, University of British Columbia, Vancouver, BC V6T 1Z4, Canada
- **Sifan Wang** - Graduate Group in Applied Mathematics and Computational Science, University of Pennsylvania, Philadelphia, PA 19104, USA
- **Paris Perdikaris** - Department of Mechanical Engineering and Applied Mechanics, University of Pennsylvania, Philadelphia, PA 19104, USA

We thank Pinaghui Huang, Zhenghao Xu, Xuening Bai, Wei Zhu, Jiequn Han, Yiwei Wang, Miles Cranmer, Chris Ormel, Hui Li, Xiaowei Jin, Shengze Cai, Bin Dong, Tie-Yan Liu, Xiaotian Gao, Wenlei Shi, Pablo Benítez-Llambay, Minhao Zhang, and Yinhao Wu for help and useful discussions in the project.

S.M. and R.D. are supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Alfred P. Sloan Foundation. S.M. and R.D. acknowledge the support of the Government of Canada's New Frontiers in Research Fund (NFRF), [NFRFE-2022-00159]. This research was enabled in part by support provided by the [Digital Research Alliance of Canada](alliance.can.ca).

[//]: # ()
[//]: # (* Hat tip to anyone whose code was used)

[//]: # (* Inspiration)

[//]: # (* etc)
