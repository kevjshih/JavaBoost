#include "singlefeaturesigmoidclassifier.h"
#include "utils.h"
#include <cmath>

using std::vector;

SingleFeatureSigmoidClassifier::
SingleFeatureSigmoidClassifier(int featColumn,
                               float threshold,
                               float smoothW,
                               float lessConf,
                               float grtrConf,
                               float dcBias) : m_featColumn(featColumn), m_threshold(threshold),
    m_lessConf(lessConf), m_grtrConf(grtrConf), m_smoothW(smoothW), m_dcBias(dcBias) {}


void SingleFeatureSigmoidClassifier::
classify(float* output, float** data, int N, int NC) {
    SingleFeatureSigmoidClassifier::classify(output, data, N, NC, m_lessConf, m_grtrConf, m_dcBias, m_smoothW, m_threshold, m_featColumn);

}

void SingleFeatureSigmoidClassifier::
classify(float* output, float ** data, int N, int NC, float lessConf, float grtrConf, float dcBias, float smoothingW, float threshold, int featColumn) {
    float alpha = grtrConf - lessConf;
    float bias = lessConf;
    for(int i = 0; i < N; ++i) {
        if(utils::isinf(data[i][featColumn])) {
            output[i] = dcBias;
        } else{
            output[i] = bias + alpha/(1+exp(-smoothingW*(data[i][featColumn]-threshold)));
        }
    }
}

vector< vector<float> > SingleFeatureSigmoidClassifier::
getParams() {
   vector< vector< float > > all_params;
   vector<float> params;
   params.push_back((float)m_featColumn);
   params.push_back(m_threshold);
   params.push_back(m_smoothW);
   params.push_back(m_lessConf);
   params.push_back(m_grtrConf);
   params.push_back(m_dcBias);
   all_params.push_back(params);
   return all_params;
}
