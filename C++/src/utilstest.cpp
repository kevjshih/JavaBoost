#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <limits>
#include "utils.h"

using std::abs;

void test_addVectorsInPlace() {
    int num = 100;
    float* a = new float[num];
    float* b = new float[num];
    for(int i = 0; i < num; ++i) {
        a[i] = i;
        b[i] = i;
    }
    utils::addVectorsInPlace(a, b, num);
    for(int i = 0; i < num; ++i) {
        if(a[i] != b[i]*2){
            printf("error in addVectorsInPlace test\n");
            exit(1);
        }
    }
    printf("addVectorsInPlace passed!\n");
    delete[] a;
    delete[] b;

    return;
}


void test_normalizeVector() {
    int num = 100;
    float* a = new float[num];
    for(int i = 0; i < num; ++i) {
        a[i] = i;
    }

    utils::normalizeVector(a, num);
    float sum = 0;

    for(int i = 0; i < num; ++i) {
        sum += a[i];
    }

    if(abs(sum - 1.0f) < 1e-4) {
        printf("normalizeVector passed!\n");
    }

    delete[] a;
}

void test_getBalancedWeights() {
    int num = 100;
    float* a = new float[num];
    int* labels =new int[num];
    for(int i = 0; i < num; ++i) {
        a[i] = i;
        if(i%2 == 0) {
            labels[i] = 1;
        } else {
            labels[i] = -1;
        }
    }
    for(int i = 0; i <  30; ++i) {
        labels[i] = 0;
    }

    utils::getBalancedWeights(a, labels, num);
    float pos_sum = 0;
    float neg_sum = 0;
    for(int i = 0; i < num; ++i) {
        if(labels[i] == 1) {
            pos_sum += a[i];
        }
        else if (labels[i] == -1) {
            neg_sum += a[i];
        }
    }
    if(abs(0.5f-pos_sum) < 1e-4 && abs(0.5f-neg_sum < 1e-4)) {
        printf("getBalancedWeights passed!\n");
    }
    delete[] a;
    delete[] labels;
}

void test_sortRowsByFirstColumn() {
    int num = 100;
    float** data = new float*[num];
    for(int i = 0; i < num; ++i) {
        data[i] = new float[3];
        data[i][0] = (float)i;
        data[i][1] = 3.0f*i;
        data[i][2] = 4.0f*i;
    }

    // sort in descending order
    utils::sortRowsByFirstColumn(data,  num, false);
    for(int i = 1; i < num; ++i) {
        if(data[i][0] > data[i-1][0]) {
            printf("sortRowsByFirstColumn failed!\n");
            return;
        }
    }

    // sort in ascending order
    utils::sortRowsByFirstColumn(data, num, true);
    for(int i = 1; i < num; ++i) {
        if(data[i][0] < data[i-1][0]) {
            printf("sortRowsByFirstColumn failed!\n");
            return;
        }
    }
    printf("sortRowsByFirstColumn passed!\n");
    for(int i = 0; i < num; ++i) {
        delete[] data[i];
    }
    delete[] data;


}

void test_isinf() {
    float val = std::numeric_limits<float>::infinity();
    bool a = utils::isinf(val);
    bool b = utils::isinf(-val);
    if(a && b) {
        printf("isinf passed!\n");
    } else{
        printf("isinf failed!\n");
    }
}

int main() {
    printf("Running test for utils functions...\n");
    test_addVectorsInPlace();
    test_normalizeVector();
    test_getBalancedWeights();
    test_sortRowsByFirstColumn();
    test_isinf();
}
