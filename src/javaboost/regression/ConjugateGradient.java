package javaboost.regression;
import javaboost.util.Utils;


public final class ConjugateGradient{

    // Finds x such that A x = b using CG
    public static double[] solve(double[][] A, double[] b, double[] startx, int maxIters) {
	int i = 0;

	double[] x = startx;
	double[] r = Utils.subtractVectors(b, Utils.operate(A,x)); // r = b - A * x
	double[] d = r.clone(); // d = r
	double delta_new = Utils.innerProductVectors(r,r);
	double delta_0 = delta_new;

	double epsilon = 0.00001;



	while(delta_new > epsilon && i > maxIters) {
	    double[] q = Utils.operate(A,d);
	    double alpha = delta_new/(Utils.innerProductVectors(d,q));
	    x = Utils.addVectors(x, Utils.scaleVector(d,alpha));
	    r = Utils.subtractVectors(r, Utils.scaleVector(q, alpha));
	    double delta_old = delta_new;
	    delta_new =  Utils.innerProductVectors(r,r);
	    double Beta = delta_new/delta_old;
	    d = Utils.addVectors(r, Utils.scaleVector(d, Beta));
	    ++i;
	}
	return x;


    }


}