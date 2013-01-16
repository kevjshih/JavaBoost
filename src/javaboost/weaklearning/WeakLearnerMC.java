package javaboost.weaklearning;

import java.util.Set;

public interface WeakLearnerMC{
    // trains on data and returns the weighted loss
    double train(final float[][] data, final int labels[], final float[] weights, final Set<Integer> pset, final Set<Integer> nset);

    WeakClassifier buildLearnedClassifier();

    float getLearnedLoss();

    int[] getTargetColumns();
}
