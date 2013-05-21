#ifndef __weaklearner_h
#define __weaklearner_h

#include "classifier.h"



class WeakLearner {
  public:
    // N = number of rows of data (applies to data, labels, weights)
    // NC = number of columns
    virtual float train(float ** data, const int* labels, const float* weights, int N, int NC) = 0;
    virtual Classifier* buildLearnedClassifier() = 0;
    virtual ~WeakLearner(){};
};


#endif
