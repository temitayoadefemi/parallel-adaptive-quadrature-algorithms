#include <math.h> 
#include <stdio.h>

double euler(double init, double step, double alpha, int numsteps)
{
   double y = init; 
   for (int i = 0; i<numsteps; i++) {
      y += step * (alpha - y); 
   } 

//   printf("alpha = %f final y = %f\n",alpha,y); 
   return y; 
}


double func1(double x) 
{
   double alpha = 100000.0 *sin(x*100000.0); 
   return euler(0.0, 0.0001, alpha, 1000); 
} 

