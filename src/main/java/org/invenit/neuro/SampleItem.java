package org.invenit.neuro;

import org.la4j.Vector;
import org.la4j.vector.dense.BasicVector;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class SampleItem {
    private Vector signals;
    private Vector expectedResult;

    public SampleItem() {
    }

    public Vector getSignals() {
        return signals;
    }

    public SampleItem setSignals(double... signals) {
        this.signals = new BasicVector(signals);
        return this;
    }

    public Vector getExpectedResult() {
        return expectedResult;
    }

    public SampleItem setExpectedResult(double... result) {
        this.expectedResult = new BasicVector(result);
        return this;
    }
}
