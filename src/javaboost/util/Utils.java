package javaboost.util;

import javaboost.conditioning.Conditional;
import java.util.List;

public class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static final float[] addVectors(float[] a, float[] b) {
	assert(a.length == b.length);

	float[] output = new float[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;

    }

    public static final double[] addVectors(double[] a, double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;
    }

    public static final float[] dotMultiplyVectors(float[] a, float[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static final double[] dotMultiplyVectors(double[] a, double[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static final double[] dotMultiplyVectors(double[] a, int[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static final float[] dotMultiplyVectors(float[] a, int[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }




    public static final void normalizeVector(float[] a) {
	float sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }


    public static final void normalizeVector(double[] a) {
	double sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }

    public static final boolean isValid(float[] features, List<Conditional> conditions) {
	for(Conditional cond : conditions) {
	    if(!cond.isValid(features)) {
		return false;
	    }
	}
	return true;

    }

}