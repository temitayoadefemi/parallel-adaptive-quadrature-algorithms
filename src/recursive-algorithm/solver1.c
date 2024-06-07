#include <math.h>
#include <omp.h>
#include <stdio.h>
#include "function.h"

// Structure to represent an interval
struct Interval {
    double left;    // Left endpoint of the interval
    double right;   // Right endpoint of the interval
    double tol;     // Tolerance for convergence
    double f_left;  // Function value at the left endpoint
    double f_mid;   // Function value at the midpoint
    double f_right; // Function value at the right endpoint
};

// Recursive function to calculate the integral using adaptive Simpson's rule
double simpson(double (*func)(double), struct Interval interval)
{
    double h = interval.right - interval.left;
    double c = (interval.left + interval.right) / 2.0;
    double d = (interval.left + c) / 2.0;
    double e = (c + interval.right) / 2.0;
    double fd = func(d);
    double fe = func(e);

    // Calculate the integral using Simpson's rule with two different step sizes
    double q1 = h / 6.0 * (interval.f_left + 4.0 * interval.f_mid + interval.f_right);
    double q2 = h / 12.0 * (interval.f_left + 4.0 * fd + 2.0 * interval.f_mid + 4.0 * fe + interval.f_right);

    // Check for convergence or minimum interval size
    if ((fabs(q2 - q1) < interval.tol) || ((interval.right - interval.left) < 1.0e-12)) {
        // Return the more accurate estimate with an error correction term
        return q2 + (q2 - q1) / 15.0;
    }
    else {
        // Divide the interval into two subintervals
        struct Interval i1, i2;
        double quad1, quad2;

        // Set up the first subinterval
        i1.left = interval.left;
        i1.right = c;
        i1.tol = interval.tol;
        i1.f_left = interval.f_left;
        i1.f_mid = fd;
        i1.f_right = interval.f_mid;

        // Set up the second subinterval
        i2.left = c;
        i2.right = interval.right;
        i2.tol = interval.tol;
        i2.f_left = interval.f_mid;
        i2.f_mid = fe;
        i2.f_right = interval.f_right;

        // Recursive calls to simpson() for the subintervals using OpenMP tasks
        #pragma omp task shared(quad1)
        quad1 = simpson(func, i1);

        #pragma omp task shared(quad2)
        quad2 = simpson(func, i2);

        // Wait for both tasks to complete
        #pragma omp taskwait
        // Return the sum of the integrals over the subintervals
        return quad1 + quad2;
    }
}

int main(void)
{
    struct Interval whole;
    double quad;

    double start = omp_get_wtime(); // Start the timer

    // Set up the initial interval
    whole.left = 0.0;
    whole.right = 10.0;
    whole.tol = 1e-06;
    whole.f_left = func1(whole.left);
    whole.f_right = func1(whole.right);
    whole.f_mid = func1((whole.left + whole.right) / 2.0);

    // Create a parallel region
    #pragma omp parallel
    {
        // Use a single thread to start the recursive calls
        #pragma omp single
        quad = simpson(func1, whole);
    }

    double time = omp_get_wtime() - start; // Calculate the elapsed time

    // Print the result and elapsed time
    printf("Result = %e\n", quad);
    printf("Time(s) = %f\n", time);
}
