# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Overview

In this project a PID controller was implemented in C++ and tuned to control the car's steering angle for driving along the track in Udacity simulator. 

<p align="center">
  <img src="./images/left_turn.gif" title = "PID Controller" alt = "PID Controller" width = "400" />
</p>
<center>PID Controller</center>

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator can be downloaded from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.



## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 



## PID controller

PID controller calculates error function (steering angle) using three components: proportional (P), derivtive (D) and integral (I)

### Proportional component (P)

Proportional component is inverse proportional to cross track error (CTE) given from the simulator. The bigger CTE is the greater is the car steering angle to compensate CTE. If the controller has only P component, then the overshooting occurs which leads to strong oscillations, as shown in the image below.

<p align="center">
  <img src="./images/only_P.gif" title = "Only P-component" alt = "Only P-component" width = "400" />
</p>
<center>Only P-component</center>

### Derivative component (D)

Derivative component considers the change rate of CTE between current and previous steps (derivative of CTE). It prevents the overshooting and reduces the oscillations: if the CTE change is small, then the steering angle is also effected less than by larger CTE change. If the car gets closer to the center of the track, the steering angle reduces and the center overshooting becomes smaller, as shown in the image below.

<p align="center">
  <img src="./images/only_PD.gif" title = "PD controller" alt = "PD controller" width = "400" />
</p>
<center>PD controller</center>

### Integral component (I)

Integral component is the sum of all previous CTE. It reduces the stady-state error so that the car returns to the center of the track, as shown in the image below.

<p align="center">
  <img src="./images/PID.gif" title = "PID controller" alt = "PD controller" width = "400" />
</p>
<center>PID controller (P = 0.2453, D = 5.55, I = 0.001)</center>


## Hyperparmeters tuning

The hyperparameters (P, I, D) of the PID controller were tuned first manually (roughly) and then using twiddle algorithm (fine tuning). Finally the tuned hyperparameters are following: P = 0.2453, D = 5.55, I = 0.001.

### Manual tuning

* First all parameters are zero
* Set P component so that the car reacts to CTE, overshoots the center of the track and starts to oscillate
* Set D component so that the car reacts to CTE properly and oscillations almost desappear
* Set I component to reduce oscillations and Stady-State error

### Twiddle

Twiddle algorithm was used for fine tuning of hyperparameters:

* The car drives full loop and the total CTE (error) is calculated
* The hyperparameters are tuned with predefined steps
* The car drives full loop again and if the total CTE reduces - this new hyperparmeters are chosen.
* The tuning steps are also increased (if CTE reduces) or reduced (if CTE doesn't reduce any more) during the algorithm
* The twiddle algorithm continues untill the sum of tunings steps is above some tollerance.

## Results

The PID controller with the parameters (P = 0.2453, D = 5.55, I = 0.001) works quite well, the car is able to drive full loop without accidents. However there are still some oscillations which the PID controller removes fast. To improve the behaviour and reduce cross track error the fine tuning of hyperparameters can be performed with smaller tollerance.
