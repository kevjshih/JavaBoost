package javaboost.weaklearning;

public class MultiFeatureLRSigmoidClassifier implements WeakClassifier{
    private double[] m_lrSolution = null;
    private double m_leftConf = 0;
    private double m_rightConf = 0;
    private int[] m_featColumns = null;
    private double m_smoothingW = 0;
    private float m_threshold  = 0;

    public MultiFeatureLRSigmoidClassifier(int[] featColumns,
					   float threshold,
					   double smoothingW,
					   double[] lrSolution,
					   double leftConf,
					   double rightConf) {
	m_featColumns = featColumns;
	m_threshold = threshold;
	m_smoothingW = smoothingW;
	m_leftConf = leftConf;
	m_rightConf = rightConf;
	m_lrSolution = lrSolution;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    boolean skip = false;
	    for(int j = 0; j < m_featColumns.length; ++j) {
		if(Float.isInfinite(data[i][m_featColumns[j]])) {
		    skip = true;
		    break;
		}
	    }
	    if(skip) {
		output[i] = 0;
	    } else {
		double rawOut = 0;
		for(int j = 0; j < m_featColumns.length; ++j) {
		    rawOut += m_lrSolution[j] * data[i][m_featColumns[j]];
		}
		rawOut += m_lrSolution[m_lrSolution.length-1]; // add bias
		double sigOut = 1/(Math.exp(-rawOut) + 1.0);
		output[i] = m_leftConf + (m_rightConf-m_leftConf)/(1+Math.exp(-m_smoothingW*(sigOut-m_threshold)));

	    }
	}
	return output;
    }

    public int[] getColumns() {
	return m_featColumns;
    }
}
