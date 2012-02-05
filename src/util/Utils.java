package util;

public class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static double[] addVectors(double[] a, double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;
    }

}