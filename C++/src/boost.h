#ifndef __boost_h
#define __boost_h


#include "weaklearner.h"
#include "additiveclassifier.h"

#include <list>

namespace boosting{

    AdditiveClassifier* train(float** data,
                             int* labels,
                             int numExamples,
                             int numColumns,
                             std::list< WeakLearner*> learners,
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
