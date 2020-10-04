#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */

  this->Kp = Kp_;
  this->Ki = Ki_;
  this->Kd = Kd_;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  PID::p_error = cte;
  PID::d_error = cte - PID::prev_cte;
  PID::prev_cte = cte;
  PID::i_error += cte;

}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return - PID::Kp * PID::p_error - PID::Kd * PID::d_error - PID::Ki * PID::i_error;
  
}


bool PID::checkLoopPassed()
{
  if(PID::steps >= PID::maxStepsInLoop)
  {
    std::cout << "Loop passed!" << std::endl;
    PID::steps = 0;
    return true;
  }
  else
  {
    //std::cout << "step: " << PID::steps << std::endl;
    PID::steps++;
    return false;
  }
  
}

void PID::accumulateError(double cte)
{
  PID::loopError += abs(cte);
}

void PID::resetLoopError()
{
  PID::loopError = 0.0;
}

void PID::setNewBestError()
{
  PID::bestErr = PID::loopError;
}

void PID::initTwiddle(bool allowTwiddle, double dKp, double dKi, double dKd, double twiddleTollerance)
{
  this->doTwiddle = allowTwiddle;
  this->dKp = dKp;
  this->dKi = dKi;
  this->dKd = dKd;
  this->twiddleTollerance = twiddleTollerance;
}


void PID::twiddle()
{
  //0 loop... accumulate error (zero-step, only on the beginning)
  switch(PID::twiddleStepCounter)
  {
    case 0:
      PID::bestErr = PID::loopError;
      break;

    case 1:
      // 1 (Kp)
      PID::Kp += PID::dKp;

      // loop (Kp)...
      break;

    case 2:
      // 2 (Kp)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        //PID::loopError = 0.0;
        PID::dKp *= 1.1;
      }
      else
      {
        PID::Kp -= 2*PID::dKp;
      }

      // loop (Kp) ...
      break;

    case 3:
      // 3 (Kp)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        dKp *= 1.1;
      }
      else
      {
        PID::Kp += PID::dKp;
        PID::dKp *= 0.9;
      }
      // loop (Kp) ...
      break;

    case 4:
      // 4 (Ki)
      PID::Ki += PID::dKi;

      // loop (Ki)...
      break;

    case 5:
      // 5 (Ki)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        //PID::loopError = 0.0;
        PID::dKi *= 1.1;
      }
      else
      {
        PID::Ki -= 2*PID::dKi;
      }

      // loop (Ki) ...
      break;

    case 6:
      // 6 (Ki)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        dKi *= 1.1;
      }
      else
      {
        PID::Ki += PID::dKi;
        PID::dKi *= 0.9;
      }
      // loop (Ki) ...
      break;

    case 7:
      // 7 (Kd)
      PID::Kd += PID::dKd;

      // loop (Kd)...
      break;

    case 8:
      // 8 (Kd)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        //PID::loopError = 0.0;
        PID::dKd *= 1.1;
      }
      else
      {
        PID::Kd -= 2*PID::dKd;
      }

      // loop (Kd) ...
      break;

    case 9:
      // 9 (Kd)
      if(PID::loopError < PID::bestErr)
      {
        PID::bestErr = PID::loopError;
        dKd *= 1.1;
      }
      else
      {
        PID::Kd += PID::dKd;
        PID::dKd *= 0.9;
      }
      break;
  }

  PID::twiddleStepCounter++;
  if(PID::twiddleStepCounter == 10)
  {
    PID::twiddleStepCounter = 1;
  }
 
  PID::iteration++;
  
  // check if twiddle has to be finished
  double dSumm = PID::dKp + PID::dKi + PID::dKd;
  if(dSumm <= PID::twiddleTollerance)
  {
    PID::doTwiddle = false;
  }

}


void PID::printInfo()
{
  std::cout << "" << std::endl;
  std::cout << "########################### printInfo() ############################" << std::endl;

  std::cout << "Iteration: " << PID::iteration << std::endl;
  std::cout << "Twiddle step: " << PID::twiddleStepCounter << std::endl;
  std::cout << "Kp: " << PID::Kp << "; Ki: " << PID::Ki << "; Kd: " << PID::Kd << std::endl;
  std::cout << "dKp: " << PID::dKp << "; dKi: " << PID::dKi << "; dKd: " << PID::dKd << "; dSumm: " << PID::dKp + PID::dKi + PID::dKd << std::endl;
  std::cout << "bestErr: " << PID::bestErr << "; loopErr: " << PID::loopError << std::endl;
  std::cout << "doTwiddle: " << PID::doTwiddle << std::endl;



  std::cout << "####################################################################" << std::endl << std::endl;
}