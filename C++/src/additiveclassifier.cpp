#include "additiveclassifier.h"
#include "utils.h"
#include "cstdio"

using std::vector;

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

vector< vector <float> > AdditiveClassifier::
getParams() {
   vector< vector <float > > all_params;
   for(std::list<Classifier*>::iterator it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {
	  vector< vector< float > > curr_params = (*it)->getParams();
	  all_params.insert(all_params.end(), curr_params.begin(), curr_params.end());
    }

   return all_params;

}

AdditiveClassifier::~AdditiveClassifier() {
    for(std::list<Classifier*>::iterator it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {
        delete (*it);
    }
}


