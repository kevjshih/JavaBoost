package javaboost.util;


import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

// This class wraps the ExecutorService instance in a singleton construct


public class ThreadPool{
    private static ExecutorService m_instance = null;
    private static int m_numThreads = 0;

    public static synchronized ExecutorService getThreadpoolInstance() {
	if (m_instance == null) {
	    m_numThreads = Math.min(1, Runtime.getRuntime().availableProcessors()-1);
	    m_instance = Executors.newFixedThreadPool(m_numThreads);
	}
	return m_instance;
    }

    public static synchronized ExecutorService getThreadpoolInstance(int numThreads) {
	if(m_instance == null) {
	    m_instance = Executors.newFixedThreadPool(numThreads);
	} else if(numThreads != m_numThreads) {
	    // if not the right number of threads
	    m_instance.shutdown(); //shutdown
	    m_instance = Executors.newFixedThreadPool(numThreads);
	}

	m_numThreads = numThreads;
	return m_instance;
    }

}
