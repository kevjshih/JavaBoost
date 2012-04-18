package javaboost.conditioning;

public final class GreaterThanColumnConditional implements Conditional{
    private int m_column;
    private int m_column2;

    public GreaterThanColumnConditional(int greaterColumn, int lesserColumn) {
	m_column = greaterColumn;
	m_column2 = lesserColumn;
    }

    public boolean isValid(final float[] val) {
	return val[m_column] > val[m_column2];
    }

    public String toString() {
	return new String("[col:"+ m_column + ">col:" +m_column2+"]");
    }

    public int[] getTargetColumns() {
	int[] columns = {m_column, m_column2};
	return columns;
    }
}
