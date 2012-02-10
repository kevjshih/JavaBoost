package javaboost.conditioning;
public final class LogicOps{
    // using bytes rather than enums to conserve space
    public static final byte AND = 0;
    public static final byte XOR = 1;

    public static String getName(byte op) {
	switch(op) {
	case AND:
	    return "AND";
	case XOR:
	    return "XOR";
	default:
	    return null;
	}
    }
}