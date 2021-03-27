#include <cmath>

#define PI 3.14159265
#define PERIOD 10

class DoubleGyrefield
{
public :

void calculateVelocity(float*location, float t, float *velocity)
{
double A = 0.1;
double w = (2*PI)/PERIOD;
double ep = 0.25;

double a_t = ep * sin(w*t);
double b_t = 1 - (2 * ep * sin(w*t));

double fx = (a_t * (location[0]*location[0])) + (b_t*location[0]);
double dfx = (2*a_t*location[0]) + b_t;

velocity[0] = (-1) * A * sin(PI*fx) * cos(PI * location[1]);
velocity[1] = PI * A * cos(PI * fx) * sin(PI * location[1]) * dfx;
}

};
