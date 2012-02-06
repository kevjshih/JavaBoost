package javaboost.conditioning;

public class GreaterThanConditional implements Conditional{
    private float m_minThresh = 0;
    private int m_column;

    public GreaterThanConditonal(float minThresh, int column) {
	m_minThresh = minThresh;
    }

    public boolean isValid(float[] val) {
	return val[m_column] > m_minThresh;
    }
}