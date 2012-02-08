package javaboost.boosting;

import java.util.List;
import javaboost.weaklearning.WeakClassifier;
import javaboost.util.Utils;

public class AdditiveClassifier{
    private List<WeakClassifier> m_classifiers = null;

    public AdditiveClassifier(List<WeakClassifier> classifiers) {
	m_classifiers = classifiers;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    output[i] = 0;
	}

	for(WeakClassifier wc: m_classifiers) {
	    output = Utils.addVectors(output, wc.classify(data));
	}

	return output;
    }

    public void listClassifiers() {
	for(int i = 0; i < m_classifiers.size(); ++i) {
	    System.out.println(m_classifiers.get(i));
	}
    }

}