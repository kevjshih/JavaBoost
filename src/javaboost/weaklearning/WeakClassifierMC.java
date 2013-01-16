package javaboost.weaklearning;
import java.io.Serializable;
import javaboost.*;

public interface WeakClassifierMC extends Serializable, ClassifierMC{
    float[] classify(float[][] data, int classId);
    int[] getColumns();
}
