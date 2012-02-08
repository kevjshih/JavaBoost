package javaboost.conditioning;

public final class GreaterThanConditional implements Conditional{
    private float m_minThresh = 0;
    private int m_column;
    private int m_column2;
    private boolean m_isRelative;

    public GreaterThanConditional(int column, float minThresh) {
	m_minThresh = minThresh;
	m_column = column;
	m_isRelative = false;
    }

    public GreaterThanCOnditional(int greaterColumn, int lesserColumn) {
	m_column = greaterColumn;
	m_column2 = lesserColumn;
	m_isRelative = true;
    }

    public boolean isValid(final float[] val) {
	if(m_isRelative) {
	    return val[m_column] > val[m_column2]
	}else {
	    return val[m_column] > m_minThresh;
	}
    }
}