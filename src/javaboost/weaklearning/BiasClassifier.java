package javaboost.weaklearning;

public class BiasClassifier implements WeakClassifier{
    private double m_bias = 0;
    public BiasClassifier(double bias) {
	m_bias = bias;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];

	for(int i = 0; i < data.length; ++i) {
	    output[i] = m_bias;
	}
	return output;
    }

    public String toString(){
	return new String("Bias: " + m_bias);
    }

}