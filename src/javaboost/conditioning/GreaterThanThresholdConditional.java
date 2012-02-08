package javaboost.conditioning;

public final class GreaterThanThresholdConditional implements Conditional{
    private float m_minThresh = 0;
    private int m_column;

    public GreaterThanThresholdConditional(int column, float minThresh) {
	m_minThresh = minThresh;
	m_column = column;

    }

    public boolean isValid(final float[] val) {
	    return val[m_column] > m_minThresh;
    }

    public String toString() {
	return new String("[col:"+ m_column + ">" + m_minThresh+"]");
    }
}