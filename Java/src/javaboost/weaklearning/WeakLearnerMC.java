package javaboost.weaklearning;

import java.util.Set;

public interface WeakLearnerMC{
    // trains on data and returns the weighted loss
    float train(final float[][] data, final int[][] labels, final float[][] weights, final Set<Integer> pset, final Set<Integer> nset);

    WeakClassifierMC buildLearnedClassifier();

    float getLearnedLoss();

    int[] getTargetColumns();
}
