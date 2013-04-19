package javaboost.weaklearning;

public class SingleFeatureSigmoidClassifier implements WeakClassifier{

    static final long serialVersionUID = -4227240934866508505L;

    private int m_featColumn;
    private float m_threshold;
    private double m_leftConf = 0;
    private double m_rightConf = 0;
    private double m_smoothW = 0;
    private double m_dcBias = 0;
    public SingleFeatureSigmoidClassifier(int featColumn,
					  float threshold,
					  double smoothW,
					  double leftConf,
					  double rightConf,
					  double dcBias) {
	m_featColumn = featColumn;
	m_threshold = threshold;
	m_leftConf= leftConf;
	m_rightConf = rightConf;
	m_smoothW = smoothW;
	m_dcBias = dcBias;
    }

    public double[] classify(float[][] data){
	double[] output = new double[data.length];
	double alpha = m_rightConf - m_leftConf;
	double bias = m_leftConf;
	for(int i = 0; i < data.length; ++i) {
	    if(Float.isInfinite(data[i][m_featColumn])) {
		output[i] = m_dcBias;
	    }
	    else{
		output[i] = bias + alpha/(1+Math.exp(-m_smoothW*(data[i][m_featColumn]-m_threshold)));
	    }

	}
	return output;
    }

    public String toString() {
	return new String("Sigmoid: "
			  +m_featColumn
			  +"@"
			  +m_threshold
			  +" SmoothingWT:"
			  +m_smoothW
			  +" less: "
			  +m_leftConf
			  +" greaterOrEqual: "
			  +m_rightConf
			  +" DC: "
			  +m_dcBias);
    }
    public double getLB() {
	return Math.min(m_dcBias, Math.min(m_leftConf, m_rightConf));
    }

    public double getUB() {
	return Math.max(m_dcBias, Math.max(m_leftConf, m_rightConf));
    }


    public int[] getColumns() {
	int[] out = {m_featColumn};
	return out;
    }

}
