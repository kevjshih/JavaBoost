package javaboost.weaklearning;

public class MultiFeatureLRClassifier implements WeakClassifier {
    private double[] m_lrSolution = null;
    private double m_negConf = 0;
    private double m_posConf = 0;
    private int[] m_featColumns = null;

    public MultiFeatureLRClassifier(int[] featColumns, double[] lrSolution,
				    double negConf, double posConf) {
	m_lrSolution = lrSolution;
	m_negConf = negConf;
	m_posConf = posConf;
	m_featColumns = featColumns;
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
		if(rawOut >= 0) {
		    output[i] = m_posConf;
		}else {
		    output[i] = m_negConf;
		}
	    }
	}

	return output;
    }

    public int[] getColumns() {
	return m_featColumns;
    }
}
