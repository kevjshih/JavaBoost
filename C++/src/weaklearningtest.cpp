#include "singlefeaturesigmoidclassifier.h"
#include "singlefeaturemultithresholdedsigmoidlearner.h"
#include "boost.h"
#include "additiveclassifier.h"
#include "utils.h"
#include "weaklearner.h"
#include "classifier.h"

#include <limits>
#include <cstdlib>
#include <cstdio>
#include <list>
#include <vector>

using std::list;
// very basic testing; basically checking for crashes and reasonable output

void test_singleFeatureSigmoidClassifier(float** data, int* labels, int N, int NC) {
    float* output = new float[N];
    SingleFeatureSigmoidClassifier::classify(output, data, N, NC, -0.4f, 0.5f, -0.1f, 1.5f, 0.5f,1);
    for(int i = 0; i < N; ++i) {
        printf("%f\n", output[i]);
    }
    delete[] output;
}


void test_boost(float** data, int* labels, int N, int NC) {


    list< WeakLearner * > learners;
    std::vector<float> threshes;
    for(int i = 0; i < 100; ++i) {
        threshes.push_back(0.8f + i*0.01);
    }

    for(int i = 0; i < NC; ++i) {
        learners.push_back( new SingleFeatureMultiThresholdedSigmoidLearner(i, threshes, 0.00001f) );
    }
    Classifier* c = boosting::train(data, labels, N, NC, learners, 2000);
    float* output = new float[N];
    c->classify(output, data, N, NC);
    delete[] output;
    delete c;

    for(std::list<WeakLearner* >::iterator it = learners.begin(); it != learners.end(); ++it) {
        delete (*it);
    }


}


int main() {
    int N = 1000;
    int NC = 40;
    float** data = new float*[N];
    int* labels = new int[N];
    srand(1);

    int pos = N/2;

    for(int i = 0; i < pos; ++i) {
        data[i] = new float[NC];
        for(int j = 0; j < NC; ++j) {
            if(rand() % 100 > 200) {
                data[i][j] = -std::numeric_limits<float>::infinity();
            } else {
                data[i][j] = (rand() %100)/100.0f + 1.1;
            }
        }
        labels[i] =  1;

    }
    for(int i = pos; i < N; ++i) {
        data[i] = new float[NC];
        for(int j = 0; j < NC; ++j) {
            if(rand() % 100 >50) {
                data[i][j] = -std::numeric_limits<float>::infinity();
            } else {
                data[i][j] = (rand() %100)/100.0f;
            }
        }
        labels[i] = -1;
    }
    test_singleFeatureSigmoidClassifier(data, labels, N, NC);
    test_boost(data, labels, N, NC);

    for(int i =0; i < N; ++i) {
        delete[] data[i];
    }
    delete[] data;
    delete[] labels;
}
