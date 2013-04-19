package javaboost.boosting;

import java.io.Serializable;
import java.util.List;

import javaboost.*;

public class LayeredClassifier implements Serializable, Classifier{
    private List<AdditiveClassifier> m_baseClassifiers = null;
    private AdditiveClassifier m_outputClassifier = null;

    public LayeredClassifier(List<AdditiveClassifier> baseClassifiers, AdditiveClassifier outputClassifier) {
	m_baseClassifiers = baseClassifiers;
	m_outputClassifier = outputClassifier;
    }

    public double[] classify(float[][] data) {
	float[][] baseOutput = new float[data.length][m_baseClassifiers.size()];
	for(int i = 0; i < m_baseClassifiers.size(); ++i) {
	    double[] output = m_baseClassifiers.get(i).classify(data);
	    for(int j = 0; j < data.length; ++j) {
		baseOutput[j][i] = (float)output[j];
	    }
	}
	return m_outputClassifier.classify(baseOutput);
    }
}