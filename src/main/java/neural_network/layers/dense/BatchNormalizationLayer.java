package neural_network.layers.dense;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.optimizer.Optimizer;
import neural_network.regularization.Regularization;
import nnarray.NNMatrix;
import nnarray.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class BatchNormalizationLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    private boolean trainable;
    @Setter
    private boolean loadWeight;

    //weight and threshold
    @Setter
    private NNMatrix weight;
    private NNMatrix derWeight;
    private NNMatrix[] optimizeWeight;
    @Setter
    private NNVector threshold;
    private NNVector derThreshold;
    private NNVector[] optimizeThreshold;

    private NNMatrix inputData;

    public BatchNormalizationLayer(int countNeuron) {
        this.countNeuron = countNeuron;
        this.trainable = true;
        initializer = new Initializer.HeNormal();
    }

    @Override
    public void initialize(int input) {
        if (!loadWeight) {
            threshold = new NNVector(countNeuron);
            weight = new NNMatrix(input, countNeuron);
            initializer.initialize(weight);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        if (optimizer.getCountParam() > 0) {
            optimizeThreshold = new NNVector[optimizer.getCountParam()];
            optimizeWeight = new NNMatrix[optimizer.getCountParam()];

            for (int i = 0; i < optimizer.getCountParam(); i++) {
                optimizeThreshold[i] = new NNVector(threshold);
                optimizeWeight[i] = new NNMatrix(weight);
            }
        }
    }

    @Override
    public void update(Optimizer optimizer) {
        if (trainable) {
            if (optimizer.getClipValue() != 0) {
                derWeight.clip(optimizer.getClipValue());
                derThreshold.clip(optimizer.getClipValue());
            }

            if (inputData.getRow() != 1) {
                derWeight.div(inputData.getRow());
                derThreshold.div(inputData.getRow());
            }

            if(regularization!= null){
                regularization.regularization(weight);
                regularization.regularization(threshold);
            }

            optimizer.updateWeight(weight, derWeight, optimizeWeight);
            optimizer.updateWeight(threshold, derThreshold, optimizeThreshold);
        }
    }

    @Override
    public void generateOutput(NNMatrix input) {
        this.inputData = input;
        outputs = inputData.mul(weight);
        outputs.addVector(threshold);
    }

    @Override
    public void generateTrainOutput(NNMatrix input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNMatrix error) {
        this.error = error.mulT(weight);
    }

    @Override
    public void generateErrorWeight(NNMatrix error) {
        if (trainable) {
            derWeight = inputData.transpose().mul(error);
            derThreshold = error.sumRow();
        }
    }

    public BatchNormalizationLayer setTrainable(boolean trainable){
        this.trainable = trainable;

        return this;
    }

    public BatchNormalizationLayer setInitializer(Initializer initializer){
        this.initializer = initializer;

        return this;
    }

    public BatchNormalizationLayer setRegularization(Regularization regularization){
        this.regularization = regularization;

        return this;
    }

    @Override
    public void info() {
        int countParam = weight.getRow() * weight.getColumn() + threshold.getLength();
        System.out.println("Dense \t\t|  " + weight.getRow() + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Dense layer\n");
        writer.write(countNeuron  + "\n");
        threshold.save(writer);
        weight.save(writer);
        if(regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static BatchNormalizationLayer read(Scanner scanner){
        BatchNormalizationLayer denseLayer = new BatchNormalizationLayer(Integer.parseInt(scanner.nextLine()));
        denseLayer.loadWeight = false;
        denseLayer.threshold = NNVector.read(scanner);
        denseLayer.weight = NNMatrix.read(scanner);
        denseLayer.setRegularization(Regularization.read(scanner));
        denseLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        return denseLayer;
    }
}
