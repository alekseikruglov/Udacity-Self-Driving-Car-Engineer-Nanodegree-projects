#include "Trajectory.h"




// Constructor

Trajectory::Trajectory(Maneuvers trjType)
{
    this->trjType = trjType;
}

Trajectory::Trajectory()
{

}

// calculates trajectory using spline
void Trajectory::calculateTrajectory(std::vector<double> ptsx, 
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
                                    std::vector<double> &map_waypoints_y)
{

    // KL, CL or CR trajectory

    // add spaced points
    std::vector<double> next_wp0;
    std::vector<double> next_wp1;
    std::vector<double> next_wp2;

    if(Trajectory::trjType == Maneuvers::KL)
    {
        
        next_wp0 = Helpers::getXY(car_s+40, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp1 = Helpers::getXY(car_s+80, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp2 = Helpers::getXY(car_s+120, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
    }
    else if(Trajectory::trjType == Maneuvers::CL)
    {
        double laneD = 2+4*lane - 4;
        next_wp0 = Helpers::getXY(car_s+50, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp1 = Helpers::getXY(car_s+100, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp2 = Helpers::getXY(car_s+150, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    }
    else if(Trajectory::trjType == Maneuvers::CR)
    {
        double laneD = 2+4*lane + 4;
        next_wp0 = Helpers::getXY(car_s+50, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp1 = Helpers::getXY(car_s+100, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
        next_wp2 = Helpers::getXY(car_s+150, laneD, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    }

    
    ptsx.push_back(next_wp0[0]);
    ptsx.push_back(next_wp1[0]);
    ptsx.push_back(next_wp2[0]);

    ptsy.push_back(next_wp0[1]);
    ptsy.push_back(next_wp1[1]);
    ptsy.push_back(next_wp2[1]);

    // coordinate system transformation (shift and rotation) so that car is in (0,0) and angle = 0 degree. To simplify math
    for(int i = 0; i < ptsx.size(); i++)
    {
        // shift car reference angle to 0 degrees
        double shift_x = ptsx[i]-ref_x;
        double shift_y = ptsy[i]-ref_y;

        ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y*sin(0-ref_yaw));
        ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y*cos(0-ref_yaw));
    }

    // create spline
    tk::spline s;

    //add points to spline

    s.set_points(ptsx, ptsy);
    
    //start with all of the previous path points from last time
    for(int i = 0; i < previous_path_x.size(); i++)
    {
        Trajectory::next_x.push_back(previous_path_x[i]);
        Trajectory::next_y.push_back(previous_path_y[i]);

        auto frenetSD = Helpers::getFrenet(previous_path_x[i], previous_path_y[i], ref_yaw, map_waypoints_x, map_waypoints_y);
        Trajectory::next_s.push_back(frenetSD[0]);
        Trajectory::next_d.push_back(frenetSD[1]);
    }

    // Calculate how to break up spline points so that we travel at our desired reference velocity
    double target_x = 60.0;
    double target_y = s(target_x);
    double target_dist = sqrt(target_x*target_x + target_y*target_y);

    double x_add_on = 0;

    // Fill up the path planner after filling it with previous points, here always 50 points
    for(int i = 1; i <= 50 - previous_path_x.size(); i++)
    {

        double N = target_dist/(0.02*ref_vel/2.24);
        
        double x_point = x_add_on + target_x/N;
        double y_point = s(x_point);

        x_add_on = x_point;

        double x_ref = x_point;
        double y_ref = y_point;

        //rotate back to normal coordinates after rotation it earlier
        x_point = x_ref*cos(ref_yaw) - y_ref*sin(ref_yaw);
        y_point = x_ref*sin(ref_yaw) + y_ref*cos(ref_yaw);


        x_point += ref_x;
        y_point += ref_y;

        Trajectory::next_x.push_back(x_point);
        Trajectory::next_y.push_back(y_point);

        // XY to Frenet (for future cost functions)
        auto frenetSD = Helpers::getFrenet(x_point, y_point, ref_yaw, map_waypoints_x, map_waypoints_y);
        Trajectory::next_s.push_back(frenetSD[0]);
        Trajectory::next_d.push_back(frenetSD[1]);

    }
}

// calculates cost for the trajectory using cost functions
void Trajectory::calculateCostForTrj(std::vector<std::vector<double>> &sensor_fusion, int currentLane, int prev_size, double car_s, double ref_yaw)
{

  //find front vehicle speed on current lane (before changing)
  double frontVehSpeedCurrentLane = Trajectory::maxSpeed*2;
  for(int i = 0; i < sensor_fusion.size(); i++)
  {
    //car is in my lane
    float d = sensor_fusion[i][6];
    if(d < (2+4*currentLane+2) && d > (2+4*currentLane-2))
    {
      frontVehSpeedCurrentLane = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);

      double check_car_s = sensor_fusion[i][5];

      check_car_s+=((double)prev_size*0.02*frontVehSpeedCurrentLane);  //update each 20 ms

      if((check_car_s > car_s) && ((check_car_s - car_s) < Trajectory::minFrontDistance+10) )
      {
        // if the car is in range -> finish search
        break;
      }    
    }
  }
  
  
    // what type of trajectory (keep lane, change left, change right)
    switch(Trajectory::trjType)
    {
        case Maneuvers::KL:
            // check distance to the front vehicle and front vehicle speed (if speed too low -> increase cost!!)
            // find ref_v to use
            for(int i = 0; i < sensor_fusion.size(); i++)
            {
                //car is in my lane
                float d = sensor_fusion[i][6];
                if(d < (2+4*currentLane+2) && d > (2+4*currentLane-2))
                {
                    double frontVehSpeed = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);

                    double check_car_s = sensor_fusion[i][5];

                    check_car_s+=((double)prev_size*0.02*frontVehSpeed);  //update each 20 ms

                    if((check_car_s > car_s) && ((check_car_s - car_s) < Trajectory::minFrontDistance) )
                    {
                        // if the car is in front of egoCar in range of minFrontDistance m and slower than maxSpeed km/h -> 
                        // increase cost (lower front veh speed -> higher cost to stay behind it)
                        Trajectory::trjCost = 1-frontVehSpeed/Trajectory::maxSpeed;
                        Trajectory::tooClose = true;
                        break;
                    }    
                }
            }
            break;
        
        
        case Maneuvers::CL:
            // avoid additional maneuvers -> set lane change cost a bit higher as keep line cost 
            Trajectory::trjCost = 0.15;

            // if the egoCar is already in left lane -> turn left not possible = max cost
            if(currentLane == 0)
            {
                Trajectory::trjCost = 1;
                break;
            }

            // check gap in the lane left (if the gap is too small - increase cost significantly)
            for(int i = 0; i < sensor_fusion.size(); i++)
            {
                // check for line left from current lane if there are vehicles and if the gap is big enough
                float d = sensor_fusion[i][6];
                if((d < (2+(currentLane-1)*4 + 2)) && d > (2+(currentLane-1)*4 - 2) )
                {
                    double closeVehSpeed = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);
                    double check_car_s = sensor_fusion[i][5];
                    check_car_s+=((double)prev_size*0.02*closeVehSpeed);  //update each 20 ms

                    // check gap in front of ego car not big enough < minFrontGap m -> max weight
                    if ((check_car_s >= car_s) && (check_car_s-car_s) < Trajectory::minFrontGap) 
                    {
                        Trajectory::trjCost = 1;
                        break;
                    }

                    // check gap back of ego car not big enough < minBackGap m -> max weight
                    if ((check_car_s <= car_s) && (car_s-check_car_s) < Trajectory::minBackGap) 
                    {
                        Trajectory::trjCost = 1;
                        break;
                    }

                    // check speed of front vehicle in changed lane (should be higher then in current lane). If slower no sense to change lane -> increase cost
                    if ( (check_car_s > car_s) && ((check_car_s-car_s) < Trajectory::minFrontGap+10) ) 
                    {                        
                        Trajectory::trjCost += 1 - exp(-(Trajectory::maxSpeed - closeVehSpeed) / (Trajectory::maxSpeed - frontVehSpeedCurrentLane));
                        //break;
                    }
                }
            }

            break;
        
        
        case Maneuvers::CR:
            // avoid additional maneuvers -> change lane change cost a bit higher as keep line cost 
            Trajectory::trjCost = 0.15;

            // if the egoCar is already in right lane -> turn right = max cost
            if(currentLane == 2)
            {
                Trajectory::trjCost = 1;
                break;
            }

            // check gap in the lane right(if the gap is too smal - increase cost significantly)
            for(int i = 0; i < sensor_fusion.size(); i++)
            {
                // check for line right from current lane if there are vehicles and if the gap is big enough
                //car is in right lane
                float d = sensor_fusion[i][6];
                if((d < (2+(currentLane+1)*4 + 2)) && d > ((2+(currentLane+1)*4 - 2)))
                {
                    double closeVehSpeed = sqrt(sensor_fusion[i][3]*sensor_fusion[i][3] + sensor_fusion[i][4]*sensor_fusion[i][4]);
                    double check_car_s = sensor_fusion[i][5];
                    check_car_s+=((double)prev_size*0.02*closeVehSpeed);  //update each 20 ms

                    // check gap in front of ego car not big enough < minFrontGap m -> max weight
                    if ((check_car_s >= car_s) && (check_car_s-car_s) < Trajectory::minFrontGap) 
                    {
                        Trajectory::trjCost = 1;
                        break;
                    }

                    // check gap back of ego car not big enough < minBackGap m -> max weight
                    if ((check_car_s <= car_s) && (car_s-check_car_s) < Trajectory::minBackGap) 
                    {
                        Trajectory::trjCost = 1;
                        break;
                    }

                    // check speed of front vehicle in changed lane (should be higher then in current lane). If slower -> no sense to change lane -> increase cost
                    if ( (check_car_s > car_s) && ((check_car_s-car_s) < Trajectory::minFrontGap+10) ) 
                    {         
                        Trajectory::trjCost += 1 - exp(-(Trajectory::maxSpeed - closeVehSpeed) / (Trajectory::maxSpeed - frontVehSpeedCurrentLane));
                        //break;
                    }
                }
            }

            break;
    }//switch

}

