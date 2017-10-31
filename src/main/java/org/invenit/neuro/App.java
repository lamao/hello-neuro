package org.invenit.neuro;

import org.apache.commons.lang3.time.StopWatch;
import org.la4j.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Hello world!
 */
public class App {

    private static int numberOfEpochs = 10000;
    private static double learningRate = 0.1;

    private static int networkSize = 5;
    private static int numberOfSignals = 4;
    private static int numberOfOutputs = 2;
    private static Sample[] trainingSample = new Sample[]{
        new Sample().setInputs(0, 0, 0, 0).setResults(0, 0),
        new Sample().setInputs(0, 0, 0, 1).setResults(0, 1),
        new Sample().setInputs(0, 0, 1, 0).setResults(0, 0),
        new Sample().setInputs(1, 0, 0, 1).setResults(1, 1),
        new Sample().setInputs(1, 0, 0, 0).setResults(1, 0),
        new Sample().setInputs(1, 0, 0, 1).setResults(1, 1),
        new Sample().setInputs(1, 0, 1, 0).setResults(1, 0),
        new Sample().setInputs(1, 1, 1, 1).setResults(1, 1)
    };

    private static Sample[] testSample = new Sample[]{
        new Sample().setInputs(1, 0, 0, 0),
        new Sample().setInputs(1, 0, 0, 1),
        new Sample().setInputs(1, 0, 1, 0),
        new Sample().setInputs(1, 0, 1, 1),
        new Sample().setInputs(1, 1, 0, 0),
        new Sample().setInputs(1, 1, 0, 1),
        new Sample().setInputs(1, 1, 1, 0),
        new Sample().setInputs(1, 1, 1, 1)
    };

    public static void main(String[] args) {

        Network network = new Network(networkSize, learningRate, numberOfSignals, numberOfOutputs);

        System.out.println("Training started");
        System.out.println(String.format("Network parameters - size=%d, learningRate=%.3f", networkSize, learningRate));
        System.out.println(String.format("Number of epochs to teach: %d", numberOfEpochs));
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        int logInterval = numberOfEpochs / 10;
        for (int i = 0; i < numberOfEpochs; i++) {
            for (Sample sample : trainingSample) {
                network.train(sample);
            }
//            network.mutate();
            if (i % logInterval == 0 || i == numberOfEpochs - 1) {
                double evaluation = 0;
                for (Sample sample : trainingSample) {
                    Vector prediction = network.predict(sample.getInputs());
                    evaluation += evaluateQuality(prediction, sample.getResults());
                }
                evaluation = evaluation / trainingSample.length;
                System.out.println(String.format("Processed %d%%, training loss is %.3f", i * 100 / numberOfEpochs, evaluation));

            }
        }
        stopWatch.stop();
//        System.out.println("Generated random was " + Arrays.toString(network.series));
//        System.out.println("Generated " + network.total + " times");
        System.out.println(String.format("Training completed. Time spent: %d ms", stopWatch.getTime(TimeUnit.MILLISECONDS)));


        for (Sample sample : testSample) {
            System.out.print("For input [");
            sample.getInputs().forEach((value) -> System.out.print(String.format("%.0f, ", value)));
            System.out.print("]");
            Vector result = network.predict(sample.getInputs());

            System.out.print("result is [");
            result.forEach((value) -> System.out.print(String.format("%.2f, ", value)));
            System.out.print("] or [");
            result.forEach((value) -> System.out.print(String.format("%.1B, ", value >= 0.5)));
            System.out.print("]");
            System.out.println();
        }
    }

    private static double evaluateQuality(Vector actualResults, Vector expectedResults) {
        return actualResults.subtract(expectedResults).transform((i, value) -> value * value).sum() / expectedResults.length();

    }

}
