package org.invenit.neuro;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.functor.VectorFunction;

import java.util.Random;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class Network {

    private int numberOfSignals;
    private int networkSize;
    private int numberOfOutputs;

    private double learningRate;

    private Matrix weight_0_1;
    private Matrix weights_1_2;

    private VectorFunction activationFunction;

    public Network(int networkSize, double learningRate, int numberOfSignals, int numberOfOutputs) {
        this.numberOfSignals = numberOfSignals;
        this.networkSize = networkSize;
        this.numberOfOutputs = numberOfOutputs;
        this.learningRate = learningRate;
        this.weight_0_1 = Basic2DMatrix.random(networkSize, numberOfSignals, new Random());
        this.weights_1_2 = Basic2DMatrix.random(numberOfOutputs, networkSize, new Random());
//        this.weight_0_1 = Basic2DMatrix.constant(networkSize, numberOfSignals, 0.5);
//        this.weights_1_2 = Basic2DMatrix.constant(numberOfOutputs, networkSize, 0.5);
        this.activationFunction = new DefaultActivationFunction();
    }

    public Vector predict(Vector inputs) {
        if (inputs.length() != numberOfSignals) {
            throw new IllegalArgumentException("Wrong input signals count");
        }

        Vector hiddenInput = weight_0_1.multiply(inputs);
        Vector hiddenOutput = hiddenInput.transform(activationFunction);

        Vector targetInput = weights_1_2.multiply(hiddenOutput);
        Vector targetOutput = targetInput.transform(activationFunction);

        return targetOutput;
    }

    public void train(Sample sample) {
        if (sample.getInputs().length() != numberOfSignals) {
            throw new IllegalArgumentException("Wrong input signals count");
        }

        if (sample.getResults().length() != numberOfOutputs) {
            throw new IllegalArgumentException("Wrong outputs count");
        }

        Vector hiddenInput = weight_0_1.multiply(sample.getInputs());
        Vector hiddenOutput = hiddenInput.transform(activationFunction);

        Vector targetInput = weights_1_2.multiply(hiddenOutput);
        Vector targetOutput = targetInput.transform(activationFunction);

        Vector errorLayer2 = targetOutput.subtract(sample.getResults());
        Vector gradientLayer2 = targetOutput.transform((i, value) -> value * (1 - value));
        Vector weightsDeltaLayer2 = errorLayer2.hadamardProduct(gradientLayer2);
        //TODO Maybe calculate after weightDelta calculation?
        weights_1_2 = weights_1_2.transform((row, column, value) -> value - hiddenOutput.get(column) * weightsDeltaLayer2.get(row) * learningRate);

        Vector errorLayer1 = weightsDeltaLayer2.multiply(weights_1_2);
        Vector gradientLayer1 = hiddenOutput.transform((i, value) -> value * (1 - value));
        Vector weightsDeltaLayer1 = errorLayer1.hadamardProduct(gradientLayer1);
        weight_0_1 = weight_0_1.transform((row, column, value) -> value - sample.getInputs().get(column) * weightsDeltaLayer1.get(row) * learningRate);
    }

//    private Random random = new Random(System.currentTimeMillis());
//    public int total = 0;
//    public int[] series = new int[10];
//    public void mutate() {
//        double rnd = random.nextDouble();
//        total++;
//        series[Long.valueOf(Math.round(rnd * 10 - 0.5)).intValue()]++;
////        if (random.nextDouble() > 0.9) {
//        if (rnd > 0.99) {
//            int row = random.nextInt(weight_0_1.rows());
//            int column = random.nextInt(weight_0_1.columns());
//            weight_0_1.set(row, column, random.nextDouble());
//        }
//    }

}
