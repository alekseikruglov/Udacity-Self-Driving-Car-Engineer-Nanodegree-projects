#ifndef PID_H
#define PID_H

#include <vector>
#include <iostream>

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  bool checkLoopPassed();
  void accumulateError(double cte);

  void resetLoopError();
  void setNewBestError();

  void twiddle();

  bool doTwiddle{true};

  void printInfo();

  void initTwiddle(bool allowTwiddle, double dKp, double dKi, double dKd, double twiddleTollerance);

 private:
  /**
   * PID Errors
   */
  double p_error{0.0};
  double i_error{0.0};
  double d_error{0.0};

  // previous CTE (for d_error calculation)
  double prev_cte{0.0};

  /**
   * PID Coefficients
   */ 
  double Kp{0.0};
  double Ki{0.0};
  double Kd{0.0};

  // for twiddle
  const int maxStepsInLoop = 1350;
  int steps{0};
  double bestErr{0.0};
  double loopError{0.0};
 
  // 10 steps
  int twiddleStepCounter{0};

  double dKp{0.05};
  double dKi{0.0005};
  double dKd{0.5};
  double twiddleTollerance{0.0};

  int iteration{0};
  

};

#endif  // PID_H