#ifndef __singlefeaturesigmoidclassifier_h
#define __singlefeaturesigmoidclassifier_h

#include "classifier.h"


class SingleFeatureSigmoidClassifier : public Classifier {
    int m_featColumn;
    float m_threshold;
    float m_lessConf;
    float m_grtrConf;
    float m_smoothW;
    float m_dcBias;

  public:
    SingleFeatureSigmoidClassifier(int featColumn,
                                   float threshold,
                                   float smoothW,
                                   float lessConf,
                                   float grtrConf,
                                   float dcBias);

    void classify(float* output, float** data, int N, int NC);

    static void classify(float* output, float ** data, int N, int NC, float lessConf, float grtrConf, float dcBias, float smoothingW, float threshold, int featColumn);


};



#endif
