#ifndef VEHICLE_H
#define VEHICLE_H

#include <iostream>
#include <string>
#include <vector>
//#include "Eigen-3.3/Eigen/Core"
//#include "Eigen-3.3/Eigen/QR"
//#include "helpers.h"
#include "spline/spline.h"

#include "helpers.h"

#include "Trajectory.h"


class Vehicle
{
private:

double ref_vel{0.5};
int lane{1};

int counter{0};

public:

std::vector<std::vector<double>> pipeline(  double car_x,
                                            double car_y,
                                            double car_yaw,
                                            double car_s, 
                                            double car_d, 
                                            double end_path_s,
                                            double end_path_d,
                                            std::vector<double> &previous_path_x,
                                            std::vector<double> &previous_path_y,
                                            std::vector<double> &map_waypoints_x, 
                                            std::vector<double> &map_waypoints_y,
                                            std::vector<double> &map_waypoints_s,
                                            std::vector<double> &map_waypoints_dx,
                                            std::vector<double> &map_waypoints_dy,
                                            std::vector<std::vector<double>> &sensor_fusion);
                                            
                                            
Trajectory findLessCostTrajectory(std::vector<Trajectory> &trajectories, double car_d, double range);

double getFrontVehicleSpeed(int currentLane, std::vector<std::vector<double>> &sensor_fusion, int prev_size, double car_s);

double getFrontVehicleDistance(int currentLane, std::vector<std::vector<double>> &sensor_fusion, int prev_size, double car_s);

bool checkEgoCarIsInLane(double car_d, double range);


};





#endif  // VEHICLE_H