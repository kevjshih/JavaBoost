package javaboost.regression;
import javaboost.util.Utils;


public final class ConjugateGradient{

    // Finds x such that A x = b using CG
    public static double[] solve(final double[][] A, final double[] b, final double[] startx, int maxIters) {
	int i = 0;

	double[] x = startx.clone();
	double[] r = Utils.subtractVectors(b, Utils.operate(A,x)); // r = b - A * x
	double[] d = r.clone(); // d = r
	double delta_new = Utils.innerProductVectors(r,r);
	double delta_0 = delta_new;

	double epsilon = 0.00001;


	while(delta_new > epsilon && i < maxIters) {
	    double[] q = Utils.operate(A,d);
	    double alpha = delta_new/(Utils.innerProductVectors(d,q));
	    Utils.addVectorsInPlace(x, Utils.scaleVector(d,alpha));
	    Utils.subtractVectorsInPlace(r, Utils.scaleVector(q, alpha));
	    double delta_old = delta_new;
	    delta_new =  Utils.innerProductVectors(r,r);
	    double Beta = delta_new/delta_old;
	    d = Utils.addVectors(r, Utils.scaleVector(d, Beta));
	    ++i;
	}
	return x;


    }


}