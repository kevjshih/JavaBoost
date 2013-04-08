package javaboost.boosting;

import java.util.List;
import java.util.HashSet;
import java.util.Set;

import javaboost.*;
import javaboost.weaklearning.*;
import javaboost.util.Utils;
import java.io.Serializable;

public class AdditiveClassifierMC implements Serializable, ClassifierMC{
    static final long serialVersionUID = 7613954747288464333L;

    private List<WeakClassifierMC> m_classifiers = null;
    private Set<Integer> m_classes = null;

    public AdditiveClassifierMC(List<WeakClassifierMC> classifiers, Set<Integer> classes) {
	m_classifiers = classifiers;
	m_classes = new HashSet<Integer>(classes);
    }

    public float[] classify(final float[][] data, int classId) {
	assert(m_classes.contains(classId));

	float[] output = new float[data.length];

	for(WeakClassifierMC wc : m_classifiers) {
	    Utils.addVectorsInPlace(output, wc.classify(data, classId));
	}
	return output;
    }

    public int[] classifyArgMax(final float[][] data) {
	int[] labels = new int[data.length];

	Integer[] classesArr = m_classes.toArray(new Integer[0]);


	for(int i = 0; i < labels.length; ++i) {
	    labels[i] = classesArr[0];
	}

	float[] bestOutput = this.classify(data, classesArr[0]);


	for(int i = 1; i < classesArr.length; ++i) {
	    float[] currOutput = this.classify(data, classesArr[i]);
	    for(int j = 0; j < currOutput.length; ++j) {
		if(currOutput[j] > bestOutput[j]) {
		    labels[j] = classesArr[i];
		    bestOutput[j] = currOutput[j];
		}
	    }

	}
	return labels;
    }

}
