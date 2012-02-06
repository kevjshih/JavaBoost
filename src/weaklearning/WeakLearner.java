package weaklearning;

public interface WeakLearner{
    // trains on data and returns the weighted loss
    double train(float[][] data, int labels[], double[] weights);

    WeakClassifier buildLearnedClassifier();
}