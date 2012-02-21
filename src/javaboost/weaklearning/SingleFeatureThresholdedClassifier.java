package javaboost.weaklearning;

public class SingleFeatureThresholdedClassifier implements WeakClassifier{
    private int m_featColumn;
    private float m_threshold;
    private double m_leftConf = 0;
    private double m_rightConf = 0;

    public SingleFeatureThresholdedClassifier(int featColumn,
					      float threshold,
					      double leftConf,
					      double rightConf) {
	m_featColumn = featColumn;
	m_threshold = threshold;
	m_leftConf= leftConf;
	m_rightConf = rightConf;

    }


    public double[] classify(float[][] data){
	double[] output = new double[data.length];

	for(int i = 0; i < data.length; ++i) {
	    if(Float.isInfinite(data[i][m_featColumn])) {
		output[i] = 0;
	    }
	    else{
		output[i] = data[i][m_featColumn] < m_threshold ? m_leftConf
		    : m_rightConf;
	    }

	}
	return output;
    }

    public String toString() {
	return new String("Threshold: "
			  +m_featColumn
			  +"@"
			  +m_threshold
			  +" less: "
			  +m_leftConf
			  +" greaterOrEqual: "
			  +m_rightConf);
    }

}