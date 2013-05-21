#include "additiveclassifier.h"
#include "utils.h"
#include "cstdio"

AdditiveClassifier::
AdditiveClassifier(std::list< Classifier* > classifiers): m_classifiers(classifiers)
{}


void AdditiveClassifier::classify(float * output, float ** data, int N, int NC) {
    for(int i = 0; i <  N; ++i) { // zero out output
        output[i]  = 0;
    }

    float* output_iter = new float[N];
    for(std::list<Classifier*>::iterator it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {



        (*it)->classify(output_iter, data, N, NC);
        utils::addVectorsInPlace(output, output_iter, N);
    }

    delete[] output_iter;
    return;
}

AdditiveClassifier::~AdditiveClassifier() {
    for(std::list<Classifier*>::iterator it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {
        delete (*it);
    }
}