package javaboost.weaklearning;
import java.io.Serializable;
import javaboost.*;

public interface WeakClassifier extends Serializable, Classifier{
    double[] classify(float[][] data);
    int[] getColumns();

}