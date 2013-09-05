#ifndef __additiveclassifier_h
#define __additiveclassifier_h

#include "classifier.h"
#include <list>
#include <vector>

class AdditiveClassifier: public Classifier {
    std::list< Classifier* > m_classifiers;
  public:
    AdditiveClassifier(std::list< Classifier* > classifiers);

    virtual void classify(float* output, float ** data, int N, int NC);

	std::vector< std::vector< float> > getParams();
    ~AdditiveClassifier();
};

#endif
