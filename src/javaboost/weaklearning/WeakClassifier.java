package javaboost.weaklearning;
import java.io.Serializable;

public interface WeakClassifier extends Serializable{
    double[] classify(float[][] data);

}