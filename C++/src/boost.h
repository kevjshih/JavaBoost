#ifndef __boost_h
#define __boost_h


#include "weaklearner.h"
#include "additiveclassifier.h"

#include <list>
#include <vector>
namespace boosting{
   // be sure to delete the classifier output
    AdditiveClassifier* train(float** data,
                             int* labels,
                             int numExamples,
                             int numColumns,
                             std::vector< WeakLearner*> learners,
                             int maxIterations);

    AdditiveClassifier* trainConcurrent(float** data,
                                       int* labels,
                                       int numExamples,
                                       int numColumns,
                                       std::list< WeakLearner * > learners,
                                        int maxIterations,
                                        int numThreads);

}

#endif
