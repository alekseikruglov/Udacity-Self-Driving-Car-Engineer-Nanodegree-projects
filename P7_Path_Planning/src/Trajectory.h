#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <iostream>
#include <string>
#include <vector>

#include "spline/spline.h"

#include "helpers.h"


// KL - keep line
// CL - change left
// CR - change right

enum class Maneuvers {KL, CL, CR};

class Trajectory
{

private:


    std::vector<double> next_x;
    std::vector<double> next_y;

    std::vector<double> next_s;
    std::vector<double> next_d;

    Maneuvers trjType;
    double trjCost{0.1};

    const double maxSpeed = 50.0;
    const double minFrontDistance = 35.0;
    const double minFrontGap = 37.0;
    const double minBackGap = 35.0;
    bool tooClose{false};

public:

    

    // Constructor

    Trajectory(Maneuvers trjType);
    Trajectory();


    // calculates trajectory using spline
    void calculateTrajectory(std::vector<double> ptsx, 
                                        std::vector<double> ptsy, 
                                        double ref_x,
                                        double ref_y,
                                        double ref_yaw,
                                        double ref_vel,
                                        std::vector<double> &previous_path_x,
                                        std::vector<double> &previous_path_y,
                                        double car_s, 
                                        int lane,
                                        std::vector<double> &map_waypoints_s,
                                        std::vector<double> &map_waypoints_x,
                                        std::vector<double> &map_waypoints_y);

    // calculates cost for the trajectory using cost functions
    void calculateCostForTrj(std::vector<std::vector<double>> &sensor_fusion, int currentLane, int prev_size, double car_s, double ref_yaw);

    // getter
    std::vector<double> &getNextX(){return Trajectory::next_x;}
    std::vector<double> &getNextY(){return Trajectory::next_y;}
    double getCost(){return Trajectory::trjCost;}
    bool getCloseFlag(){return Trajectory::tooClose;}
    string getTrjType()
    {
        switch(Trajectory::trjType)
        {
            case Maneuvers::KL:
                return "KL";
                break;

            case Maneuvers::CL:
                return "CL";
                break;

            case Maneuvers::CR:
                return "CR";
                break;

        }
        
        return "";
    }


};


#endif  // TRAJECTORY_H