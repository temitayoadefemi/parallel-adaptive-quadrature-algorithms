# Parallel Adaptive Quadrature Algorithm

This repository contains two implementations of a parallel divide-and-conquer algorithm using OpenMP. The implementations differ in their approach to task management: one uses a recursive strategy, and the other employs a LIFO (Last In, First Out) method.

## Recursive Version

The recursive version of the algorithm utilizes recursive task spawning. This approach is designed to limit the depth of task creation, optimizing task management in a multi-threaded environment. The source code is organized into several files, each tailored to specific aspects of the recursive execution.

## LIFO Version

The LIFO version implements a stack-based task management system using multiple queues. Tasks are added to the queue and processed in a LIFO order, with the most recently added task being executed first. This model can be advantageous for certain computational tasks. The source code is segmented into multiple files, each experimenting with different configurations of queues to optimize performance.

## Directory Structure

```plaintext
src/
├── recursive-algorithm/
│   └── Contains the implementation for the recursive algorithm approach.
└── lifo-algorithm/
    └── Houses the implementation for the LIFO algorithm approach.
