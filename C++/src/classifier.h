#ifndef __classifier_h
#define __classifier_h

#include <vector>

class Classifier{

  public:
    virtual void classify(float* output, float **  data, int N, int NC) = 0;
	// returns param values in order used by constructor
	virtual std::vector< std::vector<float> > getParams() = 0;
    virtual ~Classifier() {}
};


#endif
