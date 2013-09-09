#include "weaklearner.h"
#include <vector>


class SingleFeatureMultiThresholdedSigmoidLearner : public WeakLearner {
    int m_featColumn;
    std::vector<float> m_thresholds;
    int m_chosenThreshold;
    float m_lessConf;
    float m_grtrConf;
    float m_storedLoss;
    float m_dcBias;
    float m_smoothingW;

  public:
    SingleFeatureMultiThresholdedSigmoidLearner(const int featColumn, const std::vector<float> thresholds, float smoothingParam);
    virtual float  train(float ** data, const int* labels, const float* weights, int N, int NC);
    virtual Classifier* buildLearnedClassifier();

    virtual ~SingleFeatureMultiThresholdedSigmoidLearner() {}


};
