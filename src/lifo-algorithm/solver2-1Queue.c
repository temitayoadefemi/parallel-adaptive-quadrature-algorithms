#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "function.h"
#include <stdbool.h>

#define MAXQUEUE 10000

struct Interval {
    double left;    // left boundary
    double right;   // right boundary
    double tol;     // tolerance
    double f_left;  // function value at left boundary
    double f_mid;   // function value at midpoint
    double f_right; // function value at right boundary
};

struct Queue {
    struct Interval entry[MAXQUEUE]; // array of queue entries
    int top;                         // index of last entry
    omp_lock_t lock;                 // lock for synchronization
};

int total = 0;            // number of intervals ever added
int total_processed = 0;  // number of intervals processed

// Add an interval to the queue.
void enqueue(struct Interval interval, struct Queue *queue_p) {
    omp_set_lock(&(queue_p->lock));
    if (queue_p->top == MAXQUEUE - 1) {
        printf("Maximum queue size exceeded - exiting\n");
        exit(1);
    }
    queue_p->top++;
    queue_p->entry[queue_p->top] = interval;
    omp_unset_lock(&(queue_p->lock));
}

// Extract an interval from the queue.
struct Interval dequeue(struct Queue *queue_p) {
    omp_set_lock(&(queue_p->lock));
    if (queue_p->top == -1) {
        omp_unset_lock(&(queue_p->lock));
        printf("Attempt to extract from empty queue - exiting\n");
        exit(1);
    }
    struct Interval interval = queue_p->entry[queue_p->top];
    queue_p->top--;
    omp_unset_lock(&(queue_p->lock));
    return interval;
}

// Initialise the queue.
void initQueue(struct Queue *queue_p) {
    queue_p->top = -1;
    omp_init_lock(&(queue_p->lock));
}

// Check if the queue is empty.
int isempty(struct Queue *queue_p) {
    int empty;
    omp_set_lock(&(queue_p->lock));
    empty = (queue_p->top == -1);
    omp_unset_lock(&(queue_p->lock));
    return empty;
}

// The Simpson integration routine using a work queue.
double simpson(double (*func)(double), struct Queue *queue) {
    double quad = 0.0;
    bool done = false;

    // Use a pure parallel region where each thread loops until termination.
    #pragma omp parallel shared(quad, done, total, total_processed, queue)
    {
        while (!done) {
            struct Interval interval;
            bool gotWork = false;

            // Try to extract work from the queue.
            #pragma omp critical(queue_access)
            {
                if (!isempty(queue)) {
                    interval = dequeue(queue);
                    gotWork = true;
                } else if (total_processed == total) {
                    // No work in queue and all intervals have been processed.
                    done = true;
                }
            }

            if (done)
                break;

            if (!gotWork) {
                // If no work was found, yield and try again.
                #pragma omp flush(done)
                continue;
            }

            // Compute intermediate points.
            double h = interval.right - interval.left;
            double c = (interval.left + interval.right) / 2.0;
            double d = (interval.left + c) / 2.0;
            double e = (c + interval.right) / 2.0;
            double fd = func(d);
            double fe = func(e);

            // Simpson's rule estimates (3-point and 5-point).
            double q1 = h / 6.0 * (interval.f_left + 4.0 * interval.f_mid + interval.f_right);
            double q2 = h / 12.0 * (interval.f_left + 4.0 * fd + 2.0 * interval.f_mid + 4.0 * fe + interval.f_right);

            if ((fabs(q2 - q1) < interval.tol) || ((interval.right - interval.left) < 1.0e-12)) {
                // Tolerance met: add result and update processed count.
                #pragma omp atomic
                quad += q2 + (q2 - q1) / 15.0;
                #pragma omp atomic
                total_processed++;
            } else {
                // Tolerance not met: split the interval.
                struct Interval i1, i2;

                i1.left    = interval.left;
                i1.right   = c;
                i1.tol     = interval.tol;
                i1.f_left  = interval.f_left;
                i1.f_mid   = fd;
                i1.f_right = interval.f_mid;

                i2.left    = c;
                i2.right   = interval.right;
                i2.tol     = interval.tol;
                i2.f_left  = interval.f_mid;
                i2.f_mid   = fe;
                i2.f_right = interval.f_right;

                // Enqueue both new intervals.
                #pragma omp critical(queue_update)
                {
                    enqueue(i1, queue);
                    enqueue(i2, queue);
                    total += 2;
                }
            }
        } // end while
    } // end parallel region

    return quad;
}

int main(void) {
    struct Queue queue;
    initQueue(&queue);
    double start_time = omp_get_wtime();

    // Prepare the initial interval.
    struct Interval whole;
    whole.left   = 0.0;
    whole.right  = 10.0;
    whole.tol    = 1e-6;
    whole.f_left = func1(whole.left);
    whole.f_right= func1(whole.right);
    whole.f_mid  = func1((whole.left + whole.right) / 2.0);

    enqueue(whole, &queue);
    total = 1;

    // Call the Simpson integration routine.
    double result = simpson(func1, &queue);
    double elapsed_time = omp_get_wtime() - start_time;
    printf("Result = %e\n", result);
    printf("Time = %f seconds\n", elapsed_time);

    return 0;
}
