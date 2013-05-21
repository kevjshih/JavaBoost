#include "utils.h"
#include "stdlib.h"
#include <limits>

namespace utils{
    void addVectorsInPlace(float*a , float*b, int numElts) {
        for(int i =0; i < numElts; ++i) {
            a[i] = a[i] + b[i];
        }
    }


    void normalizeVector(float* a, int numElts) {
        float sum = 0;
        for(int i = 0; i < numElts; ++i) {
            sum += a[i];
        }

        for(int i =0 ; i < numElts; ++i) {
            a[i] /= sum;
        }

    }

    void getBalancedWeights(float* weights , int* labels, int numElts) {
        int numPos = 0;
        int numNeg = 0;
        for(int i = 0; i < numElts; ++i) {
            if(labels[i] == 1) {
                ++numPos;
            }else if(labels[i] == -1) {
                ++numNeg;
            }
        }

        float posWt = 0.5f/numPos;
        float negWt = 0.5f/numNeg;
        for(int i =0; i < numElts; ++i) {
            if(labels[i] == 1) {
                weights[i] = posWt;
            }else if(labels[i] == -1){
                weights[i] = negWt;
            }else {
                weights[i] = 0;
            }
        }
    }

//descending order
    int compareDescend(const void* a, const void* b) {

        if( **(float**) a < **(float**) b) {
            return 1;
        }
        else if( **(float**) a == **(float**) b) {
            return 0;
        }
        else{ // a  greater than b
                return -1;
        }
    }
// ascending order
   int compareAscend(const void* a, const void* b) {
        if( **(float**) a < **(float**) b) {
            return -1;
        }
        else if( **(float**) a == **(float**) b) {
            return 0;
        }
        else{ // a  greater than b
                return 1;
        }
    }

   void sortRowsByFirstColumn(float** data, int N, bool ascend) {
       if(ascend) {
           qsort(data, N, sizeof(float*), compareAscend);
       }else {
           qsort(data, N, sizeof(float*), compareDescend);
       }
   }

   bool isinf(float val) {
       return (val == std::numeric_limits<float>::infinity() || -val == std::numeric_limits<float>::infinity());
   }


}
