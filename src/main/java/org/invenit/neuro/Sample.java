package org.invenit.neuro;

import org.la4j.Vector;
import org.la4j.vector.dense.BasicVector;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class Sample {
    private Vector inputs;
    private Vector results;

    public Sample() {
    }

    public Vector getInputs() {
        return inputs;
    }

    public Sample setInputs(double... inputs) {
        this.inputs = new BasicVector(inputs);
        return this;
    }

    public Vector getResults() {
        return results;
    }

    public Sample setResults(double... result) {
        this.results = new BasicVector(result);
        return this;
    }
}
