
#include "Vehicle.h"



std::vector<std::vector<double>> Vehicle::pipeline( double car_x,
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
                                                    std::vector<std::vector<double>> &sensor_fusion)
{
    int prev_size = previous_path_x.size();


    if(prev_size > 0)
    {
        car_s = end_path_s;
    }

    

    // widely spaced waypoints (x,y) -> for later spline interpolation
    std::vector<double> ptsx;
    std::vector<double> ptsy;

    // reference points
    double ref_x = car_x;
    double ref_y = car_y;
    double ref_yaw = Helpers::deg2rad(car_yaw);

    // if no previous points -> use car as starting point
    if(prev_size < 2)
    {
        //two points to make tangent to the car
        double prev_car_x = car_x - cos(car_yaw);
        double prev_car_y = car_y - sin(car_yaw);

        ptsx.push_back(prev_car_x);
        ptsx.push_back(car_x);

        ptsy.push_back(prev_car_y);
        ptsy.push_back(car_y);
    }
    else
    {
        //if there are previous points -> use last previous point as reference
        ref_x = previous_path_x[prev_size-1];
        ref_y = previous_path_y[prev_size-1];
        
        //second previous point to calculate angle (yaw)
        double ref_x_prev = previous_path_x[prev_size-2];
        double ref_y_prev = previous_path_y[prev_size-2];
        ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

        //add this points
        ptsx.push_back(ref_x_prev);
        ptsx.push_back(ref_x);
        
        ptsy.push_back(ref_y_prev);
        ptsy.push_back(ref_y);
    }

    Trajectory trjCL(Maneuvers::CL);   // change left
    Trajectory trjCR(Maneuvers::CR);   // change right
    Trajectory trjKL(Maneuvers::KL);   // keep lane



    std::vector<Trajectory> trajectories = {trjCL, trjCR, trjKL};
  
    for(int i = 0; i < trajectories.size(); i++)
    {
        trajectories[i].calculateTrajectory(ptsx, ptsy, ref_x, ref_y, ref_yaw, Vehicle::ref_vel, previous_path_x, previous_path_y,
                                    car_s, Vehicle::lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

        trajectories[i].calculateCostForTrj(sensor_fusion, Vehicle::lane, prev_size, car_s, ref_yaw);
    }
    

    // find min cost trajectory
    Trajectory minCostTrj = Vehicle::findLessCostTrajectory(trajectories, car_d, 1);
    
  
    if( minCostTrj.getTrjType() == "CL")
    {
//         std::cout << "CL: " << trajectories[0].getCost() << std::endl;
//         std::cout << "CR: " << trajectories[1].getCost() << std::endl;
//         std::cout << "KL: " << trajectories[2].getCost() << std::endl;
        Vehicle::lane--;
    }

    if( minCostTrj.getTrjType() == "CR")
    { 
//         std::cout << "CL: " << trajectories[0].getCost() << std::endl;
//         std::cout << "CR: " << trajectories[1].getCost() << std::endl;
//         std::cout << "KL: " << trajectories[2].getCost() << std::endl;
        Vehicle::lane++;
    }
    
  	if(Vehicle::getFrontVehicleDistance(Vehicle::lane, sensor_fusion, prev_size, car_s) < 15.0)
    {
      // if closer 15 m to fromt car -> harder breaking
      Vehicle::ref_vel-=0.672;
    }
	else if(minCostTrj.getCloseFlag())
    {
      Vehicle::ref_vel-=0.224;
    }
  	else if( Vehicle::ref_vel < Vehicle::getFrontVehicleSpeed(Vehicle::lane, sensor_fusion, prev_size, car_s))
    {
        // if no too close cars in this trajectory -> increase speed up to 49.5
        Vehicle::ref_vel+=0.224;
    }


    std::vector<std::vector<double>> result = {minCostTrj.getNextX(), minCostTrj.getNextY()};
    
    return result;
}

Trajectory Vehicle::findLessCostTrajectory(std::vector<Trajectory> &trajectories, double car_d, double range)
{
    int idx = 0;
    double buff = 1000.0;
    for(int i = 0; i < trajectories.size(); i++)
    {
        if(trajectories[i].getCost() < buff)
        {
            buff = trajectories[i].getCost();
            idx = i;
        }
    }

    if(Vehicle::checkEgoCarIsInLane(car_d, range))
    {
        return trajectories[idx];
    }
    else
    {
        return trajectories[2];
    }
    
}

double Vehicle::getFrontVehicleSpeed(int currentLane, std::vector<std::vector<double>> &sensor_fusion, int prev_size, double car_s)
{
    for(int i = 0; i < sensor_fusion.size(); i++)
    {
      // check for line right from current lane if there are vehicles and if the gap is big enough
      //car is in right lane
      float d = sensor_fusion[i][6];
      if((d < (2+currentLane*4 + 2)) && d > ((2+currentLane*4 - 2)))
      {
        	double frontVehSpeed = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);

            double check_car_s = sensor_fusion[i][5];

            check_car_s+=((double)prev_size*0.02*frontVehSpeed);  //update each 20 ms

            if( (check_car_s > car_s) && (check_car_s - car_s) < 40.0)
            {
                // if the car is in front of egoCar in range of 30 m and slower than 50 km/h -> 
                // increase cost (lower front veh speed -> higher cost to stay behind it)
                return frontVehSpeed*2;
            }   

      }
    }
  	return 49.5;
}


bool Vehicle::checkEgoCarIsInLane(double car_d, double range)
{
    // checks if the car is in lane and maneuver is finished
    // the car stays also a short period of time in the current lane after lane change to separate maneuvers
    if((car_d < (2+Vehicle::lane *4 + range)) && car_d > ((2+Vehicle::lane*4 - range)))
    {
        Vehicle::counter++;
    }

    if(Vehicle::counter > 20)
    {
        //std::cout << "in lane! " << std::endl;
        Vehicle::counter = 0;
        return true;
    }
    else
    {
        return false;
    }
}

double Vehicle::getFrontVehicleDistance(int currentLane, std::vector<std::vector<double>> &sensor_fusion, int prev_size, double car_s)
{
    for(int i = 0; i < sensor_fusion.size(); i++)
    {
        //car is in my lane
        float d = sensor_fusion[i][6];
        if(d < (2+4*currentLane+2) && d > (2+4*currentLane-2))
        {
            double frontVehSpeed = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);

            double check_car_s = sensor_fusion[i][5];

            check_car_s+=((double)prev_size*0.02*frontVehSpeed);  //update each 20 ms

            //check svector<double> next_x_vals;
            // vector<d values greater than mine and s gap (30m)
            if(check_car_s > car_s)
            {
                // if the car is in front of egoCar in range of 30 m and slower than 50 km/h -> 
                // increase cost (lower front veh speed -> higher cost to stay behind it)
                return check_car_s - car_s;
            }    
        }
    }
  return 1000.0;
}