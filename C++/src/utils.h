#ifndef __utils_h
#define __utils_h

namespace utils{
    //a = a+b
   void addVectorsInPlace(float* a, float* b, int numElts);

   // L1 normalization of a
   void normalizeVector(float* a, int numElts);


   void getBalancedWeights(float* wts, int* labels, int numElts);

   void sortRowsByFirstColumn(float** data, int N, bool ascend);

   // returns true if float is positive or negative infinity
   bool isinf(float val);

}



#endif
