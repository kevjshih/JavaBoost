#include "boost.h"
#include "classifier.h"
#include "utils.h"
#include "cstdio"
#include <cfloat>
#include <cmath>
using std::list;
namespace boosting
{
    AdditiveClassifier* train(float** data,
                             int* labels,
                             int numExamples,
                             int numColumns,
                             std::list< WeakLearner *> learners,
                             int maxIterations) {

        list< Classifier* > output;


        float* weights = new float[numExamples];
        utils::getBalancedWeights(weights, labels, numExamples);

        float* currConfs = new float[numExamples];
        float* confs = new float[numExamples];
        float* labelDotConfs = new float[numExamples];
        for(int i = 0; i < numExamples; ++i) {
            currConfs[i] = 0;
            confs[i] = 0;
            labelDotConfs[i] = 0;
        }

        for(int t = 0; t < maxIterations; ++t) {
            WeakLearner* bestLearner = NULL;
            float bestLoss = FLT_MAX;

            for(list< WeakLearner * >::iterator it =  learners.begin(); it != learners.end(); ++it) {
                float currLoss = (*it)->train(data, labels, weights, numExamples, numColumns);
                if(currLoss < bestLoss) {
                    bestLoss = currLoss;
                    bestLearner = *it;
                }
            }
            // pick the best learner and construct corresponding classifier
            Classifier* bestClassifier = bestLearner->buildLearnedClassifier();
            output.push_back(bestClassifier);
            bestClassifier->classify(currConfs, data, numExamples, numColumns);

            // update weights
            utils::addVectorsInPlace(confs, currConfs, numExamples);

            for(int j = 0; j < numExamples; ++j) {
                labelDotConfs[j] = confs[j]*labels[j];
            }

            for(int j = 0; j < numExamples; ++j) {
                if(labels[j] != 0) {
                    weights[j] = 1/(1 + exp(labelDotConfs[j]));
                } else {
                    weights[j] = 0; // 0 weights for DCs
                }
            }
            printf("Iteration: %d weighted loss: %f\n", t, bestLoss);
            utils::normalizeVector(weights, numExamples);
        }

        delete[] confs;
        delete[] currConfs;
        delete[] labelDotConfs;
        delete[] weights;
        return new AdditiveClassifier(output);
    }

    AdditiveClassifier* trainConcurrent(float** data,
                                       int* labels,
                                       int numExamples,
                                       std::list< WeakLearner *>,
                                       int maxIterations,
                                       int numThreads) {
        return NULL;
    }

}
