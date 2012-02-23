package javaboost.regression;
import org.apache.commons.math.linear.*;


public final class ConjugateGradient{

    // Finds x such that A x = b using CG
    public static RealVector solve(RealMatrix A, RealVector b, RealVector startx, int maxIters) {
	int i = 0;

	RealVector x = startx;
	RealVector r = b.subtract(A.operate(x)); // r = b - A * x
	RealVector d = r.copy(); // d = r
	double delta_new = r.dotProduct(r);
	double delta_0 = delta_new;

	double epsilon = 0.00001;



	while(delta_new > epsilon && i > maxIters) {
	    RealVector q = A.operate(d);
	    double alpha = delta_new/(d.dotProduct(q));
	    x = x.add(d.mapMultiply(alpha));
	    r = r.subtract(q.mapMultiply(alpha));
	    double delta_old = delta_new;
	    delta_new =  r.dotProduct(r);
	    double Beta = delta_new/delta_old;
	    d = r.add(d.mapMultiply(Beta));
	    ++i;
	}
	return x;


    }


}