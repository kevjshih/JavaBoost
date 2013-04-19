package javaboost.mixturemodels;

/*
  Class for multivariate Gaussian distributions with
  diagonal covariance
  Ordering of columns, must correspond to means and sigma
 */

public final class GaussianDistribution implements Distribution{

    private double m_expCoef = -1;
    private double[] m_sigmaInverse;
    private int[] m_featColumns;
    private double[] m_means;

    public GaussianDistribution(int[] columns, double[] means, double[] sigmas) {
	assert(means.length == sigmas.length);
	assert(columns.length == means.length);

	m_featColumns = columns;
	m_sigmaInverse = new double[sigmas.length];
	double det = 1.0;
	for(int i = 0; i < columns.length; ++i) {
	    det*=sigmas[i];
	    m_sigmaInverse[i] = 1.0/sigmas[i];
	}
	m_expCoef = Math.pow(2.0*Math.PI,-columns.length/2.0)*Math.pow(det, -0.5);
	m_means = means;

    }

    public double getPDF(float[] data) {
	double exponent = 0;
	for(int i = 0; i < m_sigmaInverse.length; ++i) {
	    double deviation = data[m_featColumns[i]]-m_means[i];
	    exponent += m_sigmaInverse[i]*deviation*deviation;
	}
	exponent *=-0.5;
	return m_expCoef*Math.exp(exponent);
    }

    public int getDimensions() {
	return m_featColumns.length;
    }


}