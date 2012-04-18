package javaboost.conditioning;

public final class IsNegInfinityConditional implements Conditional{
    int m_featColumn;
    public IsNegInfinityConditional(int featColumn) {
	m_featColumn = featColumn;
    }
    public boolean isValid(float[] val) {
	if(Float.isInfinite(val[m_featColumn])) {
	    return true;
	}
	return false;
    }

    public String toString() {
	return new String("[col:"+m_featColumn + "==-Inf]");
    }

    public int[] getTargetColumns() {
	int[] columns = {m_featColumn};
	return columns;
    }
}