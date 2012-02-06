package javaboost.boosting;

import java.util.List;
import weaklearning.WeakClassifier;
import util.Utils;

public class AdditiveClassifier{
    private List<WeakClassifier> m_classifiers = null;

    public AdditiveClassifier(List<WeakClassifier> classifiers) {
	m_classifiers = classifiers;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(WeakClassifier wc: m_classifiers) {
	    output = Utils.addVectors(output, wc.classify(data));
	}

	return output;
    }

}