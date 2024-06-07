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

int total = 0;
int total_processed = 0;

// add an interval to the queue
void enqueue(struct Interval interval, struct Queue *queue_p) {
    omp_set_lock(&(queue_p->lock));
    if (queue_p->top == MAXQUEUE - 1) {
        printf("Maximum queue size exceeded - exiting\n");
        exit(1);
    }
    queue_p->top++;
    queue_p->entry[queue_p->top].left = interval.left;
    queue_p->entry[queue_p->top].right = interval.right;
    queue_p->entry[queue_p->top].tol = interval.tol;
    queue_p->entry[queue_p->top].f_left = interval.f_left;
    queue_p->entry[queue_p->top].f_mid = interval.f_mid;
    queue_p->entry[queue_p->top].f_right = interval.f_right;
    omp_unset_lock(&(queue_p->lock));
}

// extract last interval from queue
struct Interval dequeue(struct Queue *queue_p) {
    omp_set_lock(&(queue_p->lock));
    if (queue_p->top == -1) {
        omp_unset_lock(&(queue_p->lock));
        printf("Attempt to extract from empty queue - exiting\n");
        exit(1);
    }

    struct Interval interval;
    interval.left = queue_p->entry[queue_p->top].left;
    interval.right = queue_p->entry[queue_p->top].right;
    interval.tol = queue_p->entry[queue_p->top].tol;
    interval.f_left = queue_p->entry[queue_p->top].f_left;
    interval.f_mid = queue_p->entry[queue_p->top].f_mid;
    interval.f_right = queue_p->entry[queue_p->top].f_right;
    queue_p->top--;
    omp_unset_lock(&(queue_p->lock));
    return interval;
}

// initialise queue
void init(struct Queue *queue_p) {
    queue_p->top = -1;
    omp_init_lock(&(queue_p->lock));
}

// return whether queue is empty
int isempty(struct Queue *queue_p) {
    int empty;
    omp_set_lock(&(queue_p->lock));
    empty = (queue_p->top == -1);
    omp_unset_lock(&(queue_p->lock));
    return empty;
}

// get current number of queue entries
int size(struct Queue *queue_p) {
    return (queue_p->top + 1);
}

double simpson(double (*func)(double), struct Queue *queue) {
    double quad = 0.0;
    int num_threads = omp_get_max_threads();

    // Keep working until the queue is empty
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
        while (1) {
            struct Interval interval;
            int empty;
            #pragma omp critical
            {
                empty = isempty(queue);
                if (!empty) {
                    interval = dequeue(queue);
                }
            }
            if (empty) {
                break;
            }
            double h = interval.right - interval.left;
            double c = (interval.left + interval.right) / 2.0;
            double d = (interval.left + c) / 2.0;
            double e = (c + interval.right) / 2.0;
            double fd = func(d);
            double fe = func(e);

            // Calculate integral estimates using 3 and 5 points respectively
            double q1 = h / 6.0 * (interval.f_left + 4.0 * interval.f_mid + interval.f_right);
            double q2 = h / 12.0 * (interval.f_left + 4.0 * fd + 2.0 * interval.f_mid + 4.0 * fe + interval.f_right);

            if ((fabs(q2 - q1) < interval.tol) || ((interval.right - interval.left) < 1.0e-12)) {
                // Tolerance is met, add to total
                #pragma omp critical
                {
                    quad += q2 + (q2 - q1) / 15.0;
                    total_processed++;
                    if (total_processed == total) {
                        // Convergence reached and all items processed
                        printf("Convergence reached. Terminating the program.\n");
                        exit(0);
                    }
                }
            } else {
                // Tolerance is not met, split interval in two and add both halves to the queue
                struct Interval i1, i2;

                i1.left = interval.left;
                i1.right = c;
                i1.tol = interval.tol;
                i1.f_left = interval.f_left;
                i1.f_mid = fd;
                i1.f_right = interval.f_mid;

                i2.left = c;
                i2.right = interval.right;
                i2.tol = interval.tol;
                i2.f_left = interval.f_mid;
                i2.f_mid = fe;
                i2.f_right = interval.f_right;

                #pragma omp critical
                {
                    enqueue(i1, queue);
                    enqueue(i2, queue);
                    total += 2;
                }
            }
        }
    }

    return quad;
}

int main(void) {
    struct Queue queue;
    struct Interval whole;
    // Initialise queue
    init(&queue);
    double start = omp_get_wtime();
    // Add initial interval to the queue
    whole.left = 0.0;
    whole.right = 10.0;
    whole.tol = 1e-06;
    whole.f_left = func1(whole.left);
    whole.f_right = func1(whole.right);
    whole.f_mid = func1((whole.left + whole.right) / 2.0);

    enqueue(whole, &queue);
    total = 1;
    // Call queue-based quadrature routine
    double quad = simpson(func1, &queue);
    double time = omp_get_wtime() - start;
    printf("Result = %e\n", quad);
    printf("Time(s) = %f\n", time);

    return 0;
}