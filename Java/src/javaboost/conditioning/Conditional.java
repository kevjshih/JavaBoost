package javaboost.conditioning;

public interface Conditional{
    boolean isValid(float[] val);
    int[] getTargetColumns();
}