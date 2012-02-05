package boosting;

import java.util.List;
import weaklearning.WeakLearner;
import util.Utils;

public class AdditiveClassifier{
    private List<WeakLearner> m_learners = null;

    public AdditiveClassifier(List<WeakLearner> learners) {
	m_learners = learners;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(WeakLearner wl : m_learners) {
	    output = Utils.addVectors(output, wl.classify(data));
	}

	return output;
    }

}