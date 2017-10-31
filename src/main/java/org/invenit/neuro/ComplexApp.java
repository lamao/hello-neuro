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
public class ComplexApp {

    private static Logger LOGGER = LoggerFactory.getLogger(ComplexApp.class);

    private static int numberOfEpochs = 100000;
    private static double learningRate = 0.1;

    private static int networkSize = 4;
    private static int numberOfSignals = 4;
    private static int numberOfOutputs = 2;
    private static SampleItem[] trainingSample = new SampleItem[]{
        new SampleItem().setSignals(0, 0, 0, 0).setExpectedResult(0, 0),
        new SampleItem().setSignals(0, 0, 0, 1).setExpectedResult(0, 1),
        new SampleItem().setSignals(0, 0, 1, 0).setExpectedResult(0, 0),
        new SampleItem().setSignals(1, 0, 0, 1).setExpectedResult(1, 1),
        new SampleItem().setSignals(1, 0, 0, 0).setExpectedResult(1, 0),
        new SampleItem().setSignals(1, 0, 0, 1).setExpectedResult(1, 1),
        new SampleItem().setSignals(1, 0, 1, 0).setExpectedResult(1, 0),
        new SampleItem().setSignals(1, 1, 1, 1).setExpectedResult(1, 1)
    };

    private static SampleItem[] testSample = new SampleItem[]{
        new SampleItem().setSignals(1, 0, 0, 0),
        new SampleItem().setSignals(1, 0, 0, 1),
        new SampleItem().setSignals(1, 0, 1, 0),
        new SampleItem().setSignals(1, 0, 1, 1),
        new SampleItem().setSignals(1, 1, 0, 0),
        new SampleItem().setSignals(1, 1, 0, 1),
        new SampleItem().setSignals(1, 1, 1, 0),
        new SampleItem().setSignals(1, 1, 1, 1)
    };

    public static void main(String[] args) {

        ComplexNetwork network = new ComplexNetwork(networkSize, learningRate, numberOfSignals, numberOfOutputs);

        LOGGER.info("Training started.");
        LOGGER.info("Network parameters - size={}, learningRate={}", networkSize, learningRate);
        LOGGER.info("Number of epochs to teach: {}", numberOfEpochs);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        int logInterval = numberOfEpochs / 10;
        for (int i = 0; i < numberOfEpochs; i++) {
            for (SampleItem sampleItem : trainingSample) {
                network.train(sampleItem);
            }
            if (i % logInterval == 0 || i == numberOfEpochs - 1) {
                double evaluation = 0;
                for (SampleItem sampleItem : trainingSample) {
                    Vector prediction = network.predict(sampleItem.getSignals());
                    evaluation += evaluateQuality(prediction, sampleItem.getExpectedResult());
                }
                evaluation = evaluation / trainingSample.length;
                LOGGER.info("Processed {}%, training loss is {}", i * 100 / numberOfEpochs, String.format("%.3f", evaluation));

            }
        }
        stopWatch.stop();
        LOGGER.info("Training completed. Time spent: {} ms", stopWatch.getTime(TimeUnit.MILLISECONDS));


        for (SampleItem sampleItem : testSample) {
            System.out.print("For input [");
            sampleItem.getSignals().forEach((value) -> System.out.print(String.format("%.0f, ", value)));
            System.out.print("]");
            Vector result = network.predict(sampleItem.getSignals());

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
