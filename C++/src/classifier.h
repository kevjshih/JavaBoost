#ifndef __classifier_h
#define __classifier_h

class Classifier{

  public:
    virtual void classify(float* output, float **  data, int N, int NC) = 0;
    virtual ~Classifier() {}
};


#endif
