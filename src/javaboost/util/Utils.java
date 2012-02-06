package javaboost.util;

public class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static float[] addVectors(float[] a, float[] b) {
	assert(a.length == b.length);

	float[] output = new float[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;

    }

    public static double[] addVectors(double[] a, double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;
    }

    public static float[] dotMultiplyVectors(float[] a, float[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] dotMultiplyVectors(double[] a, double[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] dotMultiplyVectors(double[] a, int[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static float[] dotMultiplyVectors(float[] a, int[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }




    public static void normalizeVector(float[] a) {
	float sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }


    public static void normalizeVector(double[] a) {
	double sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }

}