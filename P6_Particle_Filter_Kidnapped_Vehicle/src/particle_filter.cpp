/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  

  this->num_particles = 30;  //  Set the number of particles to initialize

  //create a particale for initialization
  Particle newParticle;

  // Apply Gaussian distributions for x,y, theta
  std::default_random_engine gen;
	std::normal_distribution<double> xDist(x, std[0]);
	std::normal_distribution<double> yDist(y, std[1]);
	std::normal_distribution<double> thetaDist(theta, std[2]);

  // create "num_particles" initial particles
  for(int i = 0; i < num_particles; i++)
  {
    newParticle.id = i;
    newParticle.x = xDist(gen);
    newParticle.y = yDist(gen);
    newParticle.theta = thetaDist(gen);
    newParticle.weight = 1.0;

    this->particles.push_back(newParticle);
    // set weight to 1 for intial particles
    this->weights.push_back(1.0);
  }
  
  this->is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */

  //updated position and angle:
  double xF = 0.0;
  double yF = 0.0;
  double thetaF = 0.0;

  //initialize random engine for Gaussian ditribution
  std::default_random_engine gen;

  for(int i = 0; i < this->num_particles; i++)
  {
    // update position and angle
    if(yaw_rate != 0) // check division by zero
    {
      xF = this->particles[i].x + velocity*(sin(this->particles[i].theta + yaw_rate*delta_t) - sin(this->particles[i].theta))/yaw_rate;
      yF = this->particles[i].y + velocity*(cos(this->particles[i].theta) - cos(this->particles[i].theta + yaw_rate*delta_t))/yaw_rate;
    }
    else
    {
      xF = this->particles[i].x;
      yF = this->particles[i].y;
    }
    
    thetaF = this->particles[i].theta + yaw_rate*delta_t;

    // add random Gaussian noise
    std::normal_distribution<double> xDist(xF, std_pos[0]);
    std::normal_distribution<double> yDist(yF, std_pos[1]);
    std::normal_distribution<double> thetaDist(thetaF, std_pos[2]);
    
    this->particles[i].x = xDist(gen);
    this->particles[i].y = yDist(gen);
    this->particles[i].theta = thetaDist(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  
  for(unsigned int i = 0; i < observations.size(); i++)
  {
    vector<double> distances;
    for(LandmarkObs &thePrediction : predicted)
    {
      // save in this vector all distances from thePrediction to each observation
      distances.push_back(HELPER_FUNCTIONS_H_::dist(thePrediction.x,thePrediction.y, observations[i].x, observations[i].y));
    }
    // find index of the lower distance
    int minDistanceIndex = std::min_element(distances.begin(),distances.end()) - distances.begin();

    // write ID of closest prediction to the observation
    observations[i].id = predicted[minDistanceIndex].id;
  }
  
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. 
   * 
   *   The observations are given in the VEHICLE'S coordinate system -> transformation
   *    to MAP's coordinate system is needed. 
   */

  // update weights of each particle

  //for weights normalization
  double weightSumNormalizer = 0.0;
  
  for(unsigned int i = 0; i < this->particles.size(); i++)
  {

    //find all landmarks within sensor range
    vector<LandmarkObs> landmarksWithinSensorRange;
    for(unsigned int n = 0; n < map_landmarks.landmark_list.size(); n++)
    {
      LandmarkObs okLandMark = {map_landmarks.landmark_list[n].id_i,0,0};
      if(HELPER_FUNCTIONS_H_::dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[n].x_f, map_landmarks.landmark_list[n].y_f) <= sensor_range)
      {
        
        okLandMark.x = map_landmarks.landmark_list[n].x_f;
        okLandMark.y = map_landmarks.landmark_list[n].y_f;
        okLandMark.id = map_landmarks.landmark_list[n].id_i;
        landmarksWithinSensorRange.push_back(okLandMark);
      }    
    }

    // transform observations from car to map coordinates
    vector<LandmarkObs> observationsInMapCoords;
    double xMapObs = 0.0;
    double yMapObs = 0.0;
    int idObs = 0;
    for(unsigned int j = 0; j < observations.size(); j++)
    {
      xMapObs = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      yMapObs = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      idObs = observations[j].id;

      LandmarkObs transformedCoordObs;
      transformedCoordObs.x = xMapObs;
      transformedCoordObs.y = yMapObs;
      transformedCoordObs.id = idObs;

      observationsInMapCoords.push_back(transformedCoordObs);
    }

    // associate nearest predictions to observations
    ParticleFilter::dataAssociation(landmarksWithinSensorRange, observationsInMapCoords);

    // save associations for debugging
    ParticleFilter::SetAssociations(particles[i], particles[i].associations, particles[i].sense_x, particles[i].sense_y);

    // calculate weights with multi-variant Gaussian distribution 
    //go through all observations for this particle
    for(unsigned int n = 0; n < observationsInMapCoords.size(); n++)
    {
      
      // landmark corresponding to associated observation index
      Map::single_landmark_s slm = ParticleFilter::getLandmarkForObservation(observationsInMapCoords[n], map_landmarks);
      particles[i].weight *= ParticleFilter::multivProb(std_landmark[0], std_landmark[1], 
                                                  observationsInMapCoords[n].x, observationsInMapCoords[n].y,
                                                  slm.x_f, 
                                                  slm.y_f);
    }

    // save total weight of a particle:
    this->weights[i] = particles[i].weight;

    //calculazte the sum of all particle weights to make normalization
    weightSumNormalizer += particles[i].weight;
  }

  //normalize particles weights
  for(unsigned int i = 0; i < this->particles.size(); i++)
  {
    if(weightSumNormalizer != 0.0)
    {
      this->particles[i].weight /= weightSumNormalizer;
      this->weights[i] = particles[i].weight;
    }
    else
    {
      this->particles[i].weight = 1.0;
      this->weights[i] = 1.0;
    }
  }
}

void ParticleFilter::resample() {
  /**
   *  Resample particles with replacement with probability proportional 
   *   to their weight. 
   */

  // Resampling wheel

  // find max weight
  double weightMax = *std::max_element(this->weights.begin(), this->weights.end()); // max_element() returns iterator -> dereference!

  // random distribution:
  std::uniform_real_distribution<double> doubleDist(0.0, 1.0);  //for beta-update
  std::uniform_int_distribution<int> intDist(0, num_particles - 1); //for index random initialization
  std::default_random_engine gen;

  int index = intDist(gen);

  double beta = 0.0;
  vector<Particle> resampledParticles;
  for(unsigned int i = 0; i < this->particles.size(); i++) {
    beta += doubleDist(gen) * 2.0 * weightMax;

    while(beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  this->particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


// calculates particle weight using multivariant Gaussian distribution
double ParticleFilter::multivProb(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y) 
{
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

// finds corresponding landmark for this observation by id
Map::single_landmark_s ParticleFilter::getLandmarkForObservation(const LandmarkObs &currentObservation, const Map &map_landmarks)
{
  Map::single_landmark_s resultLm = {0, 0.0, 0.0};

  for(unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++)
  {
    if(currentObservation.id == map_landmarks.landmark_list[i].id_i)
    {
      resultLm.id_i = map_landmarks.landmark_list[i].id_i;
      resultLm.x_f = map_landmarks.landmark_list[i].x_f;
      resultLm.y_f = map_landmarks.landmark_list[i].y_f;    
    }
  }
  return resultLm;
}