//
// Created by Dennis-Florian Herr on 13/06/2022.
//

#include <string>
#include <deque>
#include <future>
#include <functional>

#include "Utility.h"
#include <thread>
#include <mutex>
#include <iostream>

#define MEASURE_TIME true

const int NUM_THREADS = 30;
std::mutex mutex;

struct Problem {
    Sha1Hash sha1_hash;
    int problemNum;
};


/*
 * TODO@Students: Implement a thread safe queue.
 * Tip: use a condition variable to make threads wait when the queue is empty and there is nothing to pop().
 * https://en.cppreference.com/w/cpp/thread/condition_variable
 */
class ProblemQueue {
    public:
        void push(Problem problem){
            problem_queue.push_back(problem);
        }

        Problem pop(){
            Problem p = problem_queue.front();
            problem_queue.pop_front();
            return p;
        }

        bool empty(){
            return problem_queue.empty();
        }

    private:
        std::deque<Problem> problem_queue;

};

ProblemQueue problemQueue;


// generate numProblems sha1 hashes with leadingZerosProblem leading zero bits
// This method is intentionally compute intense so you can already start working on solving
// problems while more problems are generated
void generateProblem(int seed, int numProblems, int leadingZerosProblem){
    srand(seed);

    for(int i = 0; i < numProblems; i++){
        std::string base = std::to_string(rand()) + std::to_string(rand());
        Sha1Hash hash = Utility::sha1(base);
        do{
            // we keep hashing ourself until we find the desired amount of leading zeros
            hash = Utility::sha1(hash);
        }while(Utility::count_leading_zero_bits(hash) < leadingZerosProblem);
        problemQueue.push(Problem{hash, i});
    }
}

// This method repeatedly hashes itself until the required amount of leading zero bits is found
Sha1Hash findSolutionHash(Sha1Hash hash, int leadingZerosSolution){
    do{
        // we keep hashing ourself until we find the desired amount of leading zeros
        hash = Utility::sha1(hash);
    }while(Utility::count_leading_zero_bits(hash) < leadingZerosSolution);

    return hash;
}

void solveProblem(Sha1Hash* solutionHashes, const int leadingZerosSolution){
    while(1) {
        mutex.lock();
        if(!problemQueue.empty()){
            Problem p = problemQueue.pop();
            mutex.unlock();
            solutionHashes[p.problemNum] = findSolutionHash(p.sha1_hash, leadingZerosSolution);
        }else{
            mutex.unlock();
            break;
        }
    }
}


int main(int argc, char *argv[]) {
    int leadingZerosProblem = 8;
    int leadingZerosSolution = 11;
    int numProblems = 10000;

    //Not interesting for parallelization
    Utility::parse_input(numProblems, leadingZerosProblem, leadingZerosSolution, argc, argv);
    Sha1Hash solutionHashes[numProblems];
    Sha1Hash* solutionHashesPtr = solutionHashes;

    std::thread solveThreads[NUM_THREADS];

    unsigned int seed = Utility::readInput();

    #if MEASURE_TIME
    struct timespec generation_start, generation_end;
    clock_gettime(CLOCK_MONOTONIC, &generation_start);
    #endif

    /*
    * TODO@Students: Generate the problem in another thread and start already working on solving the problems while the generation continues
    */
    std::thread generateThread = std::thread(generateProblem, seed, numProblems, leadingZerosProblem);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    #if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &generation_end);
    double generation_time = (((double) generation_end.tv_sec + 1.0e-9 * generation_end.tv_nsec) - ((double) generation_start.tv_sec + 1.0e-9 * generation_start.tv_nsec));
    fprintf(stderr, "Generate Problem time:  %.7gs\n", generation_time);

    struct timespec solve_start, solve_end;
    clock_gettime(CLOCK_MONOTONIC, &solve_start);
    #endif

    /*
    * TODO@Students: Create worker threads that parallelize this functionality. Add the synchronization directly to the queue
    */

    for(int i=0; i < NUM_THREADS; i++){
        solveThreads[i] = std::thread(solveProblem, solutionHashesPtr, leadingZerosSolution);
    }

    generateThread.join();
    for(int i=0; i < NUM_THREADS; i++){
        solveThreads[i].join();
    }
    #if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &solve_end);
    double solve_time = (((double) solve_end.tv_sec + 1.0e-9 * solve_end.tv_nsec) - ((double) solve_start.tv_sec + 1.0e-9 * solve_start.tv_nsec));
    fprintf(stderr, "Solve Problem time:     %.7gs\n", solve_time);
    #endif

    /*
    * TODO@Students: Make sure all work has finished before calculating the solution
    * Tip: Push a special problem for each thread onto the queue that tells a thread to break and stop working
    */

    Sha1Hash solution;
    // guarantee initial solution hash data is zero
    memset(solution.data, 0, SHA1_BYTES);
    // this doesn't need parallelization. it's neglectibly fast
    for(int i = 0; i < numProblems; i++){
        solution = Utility::sha1(solution, solutionHashes[i]);
    }

    Utility::printHash(solution);
    printf("DONE\n");

    return 0;
}
