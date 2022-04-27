package neural_network.layers.dense;

import lombok.Setter;
import neural_network.optimizer.Optimizer;
import neural_network.regularization.Regularization;
import nnarray.NNMatrix;
import nnarray.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class BatchNormalizationLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private boolean trainable;
    @Setter
    private boolean loadWeight;

    private final float momentum;

    //betta
    @Setter
    private NNVector betta;
    private NNVector derBetta;
    private NNVector[] optimizeBetta;
    //gamma
    @Setter
    private NNVector gamma;
    private NNVector derGamma;
    private NNVector[] optimizeGamma;

    private NNMatrix inputData;

    private NNVector movingMean;
    private NNVector movingVar;

    private NNVector mean, var;
    private NNMatrix normOutput;

    public BatchNormalizationLayer() {
        this(0.99);
    }

    public BatchNormalizationLayer(double momentum) {
        this.momentum = (float) momentum;
        this.trainable = true;
    }

    @Override
    public void initialize(int input) {
        this.countNeuron = input;
        if (!loadWeight) {
            gamma = new NNVector(countNeuron);
            movingMean = new NNVector(countNeuron);
            movingVar = new NNVector(countNeuron);
            betta = new NNVector(countNeuron);
            gamma.fill(1);
            movingVar.fill(1);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        if (optimizer.getCountParam() > 0) {
            optimizeGamma = new NNVector[optimizer.getCountParam()];
            optimizeBetta = new NNVector[optimizer.getCountParam()];

            for (int i = 0; i < optimizer.getCountParam(); i++) {
                optimizeGamma[i] = new NNVector(gamma);
                optimizeBetta[i] = new NNVector(betta);
            }
        }
    }

    @Override
    public void update(Optimizer optimizer) {
        if (trainable) {
            if (optimizer.getClipValue() != 0) {
                derBetta.clip(optimizer.getClipValue());
                derGamma.clip(optimizer.getClipValue());
            }

            if (inputData.getRow() != 1) {
                derBetta.div(inputData.getRow());
                derGamma.div(inputData.getRow());
            }

            if (regularization != null) {
                regularization.regularization(betta);
                regularization.regularization(gamma);
            }

            optimizer.updateWeight(betta, derBetta, optimizeBetta);
            optimizer.updateWeight(gamma, derGamma, optimizeGamma);
        }
    }

    @Override
    public void generateOutput(NNMatrix input) {
        normOutput = input.normalization(movingMean, movingVar);
        outputs = normOutput.mulAndAdd(gamma, betta);
    }

    @Override
    public void generateTrainOutput(NNMatrix input) {
        inputData = input;
        mean = input.sumRow();
        mean.div(input.getRow());
        var = input.sumRowSubPow2(mean);
        var.div(input.getRow());

        movingMean.momentum(mean, momentum);
        movingVar.momentum(var, momentum);

        normOutput = input.normalization(mean, var);
        outputs = normOutput.mulAndAdd(gamma, betta);
    }

    @Override
    public void generateError(NNMatrix error) {
        NNMatrix errorNormOut = error.mul(gamma);
        NNVector errorVar = error.derVar(inputData, mean, var, gamma);
        NNVector errorMean = error.derMean(inputData, mean, var, errorVar, gamma);

        this.error = errorNormOut.derNorm(inputData, mean, errorMean, var, errorVar);
    }

    @Override
    public void generateErrorWeight(NNMatrix error) {
        if (trainable) {
            derBetta = error.sumRow();
            derGamma = error.mulAndSumRow(normOutput);
        }
    }

    public BatchNormalizationLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public BatchNormalizationLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public void info() {
        int countParam = betta.getLength() * 4;
        System.out.println("Batch norm\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Batch normalization layer\n");
        writer.write(momentum + "\n");
        gamma.save(writer);
        betta.save(writer);
        movingMean.save(writer);
        movingVar.save(writer);

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static BatchNormalizationLayer read(Scanner scanner) {
        BatchNormalizationLayer layer = new BatchNormalizationLayer(Float.parseFloat(scanner.nextLine()));
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.movingMean = NNVector.read(scanner);
        layer.movingVar = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}
