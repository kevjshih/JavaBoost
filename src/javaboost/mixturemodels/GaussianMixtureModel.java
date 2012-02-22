package javaboost.mixturemodels;

import java.util.List;

public final class GaussianMixtureModel implements Distribution{

    private List<GaussianDistribution> m_distributions = null;
    private double[] m_mixtureProbabilities;

    public GaussianMixtureModel(List<GaussianDistribution> dists, double[] mixtureProbabilities) {
	m_distributions = dists;
	m_mixtureProbabilities = mixtureProbabilities;
    }

    public double getPDF(float[] data) {
	double prob = 0;
	for(int i = 0; i < m_distributions.size(); ++i) {
	    prob += m_mixtureProbabilities[i]*m_distributions.get(i).getPDF(data);
	}
	return prob;
    }

}