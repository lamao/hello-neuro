package org.invenit.neuro;

import org.la4j.vector.functor.VectorFunction;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class DefaultActivationFunction implements VectorFunction {
    @Override
    public double evaluate(int i, double value) {
        return 1 / (1 + Math.exp(-value));
    }
}
