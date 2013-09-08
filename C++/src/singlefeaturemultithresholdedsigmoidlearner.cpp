#include "singlefeaturemultithresholdedsigmoidlearner.h"
#include "singlefeaturesigmoidclassifier.h"
#include "utils.h"
#include <cmath>
#include <cstdio>
#include <cfloat>
using std::vector;

// constructor
SingleFeatureMultiThresholdedSigmoidLearner::
SingleFeatureMultiThresholdedSigmoidLearner(
    const int featColumn, const vector<float> thresholds,
    float smoothingParam):m_featColumn(featColumn), m_thresholds(thresholds), m_smoothingW(smoothingParam) {}


// train function
float SingleFeatureMultiThresholdedSigmoidLearner::
train(float ** data, const int* labels, const float* weights, int N, int NC) {

    // some initalization for safety
    m_lessConf = 0;
    m_grtrConf = 0;
    m_dcBias = 0;
	m_storedLoss = 1000;

    float regularizer = 1.0f/N;
    float** dataLabelsSorted = new float*[N];
    for(int i = 0; i < N; ++i) {
        dataLabelsSorted[i] = new float[3];
        dataLabelsSorted[i][0] = data[i][m_featColumn];
        dataLabelsSorted[i][1] = labels[i];
        dataLabelsSorted[i][2] = weights[i];
    }

    // sort dataLobelsSorted by first column in ascending order
    utils::sortRowsByFirstColumn(dataLabelsSorted,  N, true);

    int numBins = m_thresholds.size() +1;
    float* cumPosBins = new float[numBins];
    float* cumNegBins = new float[numBins];
    for(int i = 0; i < numBins; ++i) {
        cumPosBins[i] = 0;
        cumNegBins[i] = 0;
    }
    float dcPosWeights = 0;
    float dcNegWeights = 0;

    // compute the weights
    unsigned int binIdx = 0;
    for(int i = 0; i < N; ++i) {
        // inf case
        if(utils::isinf(dataLabelsSorted[i][0])){
            if(dataLabelsSorted[i][1] >= 0) {
                dcPosWeights+= dataLabelsSorted[i][2];
            }else{
                dcNegWeights+= dataLabelsSorted[i][2];
            }
            continue;
        }
        // update bins
        while(binIdx < m_thresholds.size() &&
              dataLabelsSorted[i][0] >= m_thresholds[binIdx]) {
            ++binIdx;
        }
        if(dataLabelsSorted[i][1] >= 0) {
            cumPosBins[binIdx]+= dataLabelsSorted[i][2];
        }else{
            cumNegBins[binIdx]+= dataLabelsSorted[i][2];
        }
    }
    m_dcBias = 0.5f*log((regularizer+ dcPosWeights)/(regularizer+ dcNegWeights));

    // compute the cumsum
    for(int i = 1; i < numBins; ++i) {
        cumPosBins[i] += cumPosBins[i-1];
        cumNegBins[i] += cumNegBins[i-1];
    }
    float posSum = cumPosBins[numBins-1];
    float negSum = cumNegBins[numBins-1];

    float* lessConfs = new float[m_thresholds.size()];
    float* grtrConfs = new float[m_thresholds.size()];

    float* output = new float[N];

    float loss = 0;
    float bestLoss = FLT_MAX;
    int bestThresh = -1;
    // evaluate on training data and compute loss for each thresh point
    for(unsigned int t = 0; t < m_thresholds.size(); ++t) {
        float grtrPos = posSum - cumPosBins[t];
        float grtrNeg = negSum - cumNegBins[t];
        lessConfs[t] = 0.5f*log((regularizer+cumPosBins[t])/(regularizer+cumNegBins[t]));
        grtrConfs[t] = 0.5f*log((regularizer+grtrPos)/(regularizer+grtrNeg));

        // static function call
        SingleFeatureSigmoidClassifier::classify(output, dataLabelsSorted, N, 3, lessConfs[t], grtrConfs[t], m_dcBias, m_smoothingW, m_thresholds[t], 0);
        loss = 0;
        // compute the logistic loss for threshold
        for(int i = 0; i < N; ++i) {
            loss += dataLabelsSorted[i][2]*log(1+exp(-dataLabelsSorted[i][1]*output[i]));
        }

        if(loss < bestLoss) {
            bestThresh = t;
            bestLoss = loss;
        }

    }

    // store best params
    m_storedLoss = bestLoss;
    m_chosenThreshold = bestThresh;
    m_lessConf = lessConfs[bestThresh];
    m_grtrConf =  grtrConfs[bestThresh];


    // memory cleanup
    for(int i = 0; i < N; ++i) {
        delete[] dataLabelsSorted[i];
    }
    delete[] lessConfs;
    delete[] grtrConfs;
    delete[] dataLabelsSorted;
    delete[] cumPosBins;
    delete[] cumNegBins;
    delete[] output;

    // return best loss
    return m_storedLoss;

}

Classifier* SingleFeatureMultiThresholdedSigmoidLearner::
buildLearnedClassifier() {
    return new SingleFeatureSigmoidClassifier(m_featColumn,
                                              m_thresholds[m_chosenThreshold],
                                              m_smoothingW,
                                              m_lessConf,
                                              m_grtrConf,
                                              m_dcBias);
}
