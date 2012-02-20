package javaboost.weaklearning;

public class BiasLearner implements WeakLearner{
    private double m_bias = 0;
    private double m_storedLoss = 0;
    public BiasLearner(){}

    public final double train(final float[][] data, final int labels[], final double[] weights) {
	double weightPos = 0;
	double weightNeg = 0;
	for(int i = 0; i < labels.length; ++i) {
	    if(labels[i] >= 0) {
		weightPos += weights[i];
	    }else{
		weightNeg += weights[i];
	    }
	}

	m_bias = 0.5*Math.log(weightPos/weightNeg);
	m_storedLoss = Math.exp(-m_bias)*weightPos + Math.exp(m_bias)*weightNeg;
	m_storedLoss = m_storedLoss/(1+m_storedLoss);
	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new BiasClassifier(m_bias);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

}