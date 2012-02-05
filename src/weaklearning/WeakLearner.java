package weaklearning;

public interface WeakLearner{
    void train(float[][] data, int labels[]);
    int[] classify(float[][] data);
}