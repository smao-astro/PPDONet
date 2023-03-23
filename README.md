# PPDONet: Deep Operator Networks for Fast Prediction of Solutions in Disk-Planet Systems

We develop a tool, which we name Protoplanetary Disk Operator Network (PPDONet), that can predict the solution of disk-planet interactions in protoplanetary disks in real-time. We base our tool on Deep Operator Networks (DeepONets), a class of neural networks capable of learning non-linear operators to represent deterministic and stochastic differential equations.
With PPDONet we map three scalar parameters in a disk-planet system -- the Shakura & Sunyaev viscosity $\alpha$, the disk aspect ratio $h_0$, and the planet-star mass ratio $q$ -- to steady-state solutions of the disk surface density, radial velocity, and azimuthal velocity.  We demonstrate the accuracy of the PPDONet solutions using a comprehensive set of tests. Our tool is able to predict the outcome of disk-planet interaction for one system in less than a second on a laptop, speeding the calculation by many orders of magnitude compared with conventional numerical solvers. A public implementation of PPDONet is available at \url{https://github.com/smao-astro/PPDONet}.

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

### Without Data

```commandline
python -m visualization.real_time_prediction_single_var --run_dir trained_network/single_log_sigma
```

## Built With

## Contributing

## Versioning

## Authors

* **Shunyuan Mao** - *Initial work* - [smao-astro](https://github.com/smao-astro/PPDONet)

[//]: # ()
[//]: # (See also the list of [contributors]&#40;https://github.com/your/project/contributors&#41; who participated in this project.)

## License

[//]: # ()
[//]: # (This project is licensed under the MIT License - see the [LICENSE.md]&#40;LICENSE.md&#41; file for details)

## Acknowledgments

[//]: # ()
[//]: # (* Hat tip to anyone whose code was used)

[//]: # (* Inspiration)

[//]: # (* etc)
