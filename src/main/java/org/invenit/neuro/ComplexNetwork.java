package org.invenit.neuro;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.functor.VectorFunction;

import java.util.Random;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class ComplexNetwork {

    private int numberOfSignals;
    private int networkSize;
    private int numberOfOutputs;
    private double learningRate;

    private Matrix weight_0_1;
    private Matrix weights_1_2;

    private VectorFunction activationFunction;

    public ComplexNetwork(int networkSize, double learningRate, int numberOfSignals, int numberOfOutputs) {
        this.networkSize = networkSize;
        this.learningRate = learningRate;
        this.numberOfSignals = numberOfSignals;
        this.numberOfOutputs = numberOfOutputs;
//        this.weight_0_1 = Basic2DMatrix.random(networkSize, numberOfSignals, new Random());
//        this.weights_1_2 = Basic2DMatrix.random(numberOfOutputs, networkSize, new Random());
        this.weight_0_1 = Basic2DMatrix.constant(networkSize, numberOfSignals, 0.5);
        this.weights_1_2 = Basic2DMatrix.constant(numberOfOutputs, networkSize, 0.5);
        this.activationFunction = new DefaultActivationFunction();
    }

    public Vector predict(Vector inputs) {
        if (inputs.length() != numberOfSignals) {
            throw new IllegalArgumentException("Wrong input signals count");
        }

        Vector hiden_input = weight_0_1.multiply(inputs);
//        System.out.println("Hiden weight_0_1 " + hiden_input);

        Vector hiden_output = hiden_input.transform(activationFunction);
//        System.out.println("Hiden output " + hiden_output);

        Vector output = weights_1_2.multiply(hiden_output);
//        System.out.println("Output " + output);

        return output.transform(activationFunction);
    }

    public void train(SampleItem sample) {
        if (sample.getSignals().length() != numberOfSignals) {
            throw new IllegalArgumentException("Wrong input signals count");
        }

        if (sample.getExpectedResult().length() != numberOfOutputs) {
            throw new IllegalArgumentException("Wrong outputs count");
        }

        Vector inputs_1 = weight_0_1.multiply(sample.getSignals());
        Vector outputs_1 = inputs_1.transform(activationFunction);

        Vector inputs2 = weights_1_2.multiply(outputs_1);
        Vector outputs2 = inputs2.transform(activationFunction);
        Vector actualResult = outputs2;

        Vector errorLayer2 = actualResult.subtract(sample.getExpectedResult());
        Vector gradientLayer2 = actualResult.transform((i, value) -> value * (1 - value));
        Vector weigthsDeltaLayer2 = errorLayer2.hadamardProduct(gradientLayer2);
        weights_1_2 = weights_1_2.transform((i, k, value) -> value - outputs_1.get(k) * weigthsDeltaLayer2.get(i) * learningRate);

        Vector errorLayer1 = weigthsDeltaLayer2.multiply(weights_1_2);
        Vector gradientLayer1 = outputs_1.transform((i, value) -> value * (1 - value));
        Vector weights_delta_layer_1 = errorLayer1.hadamardProduct(gradientLayer1);
        weight_0_1 = weight_0_1.transform((i, k, value) -> value - sample.getSignals().get(k) * weights_delta_layer_1.get(i) * learningRate);
    }

}
