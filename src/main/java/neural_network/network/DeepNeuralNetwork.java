package neural_network.network;

import neural_network.layers.dense.DenseNeuralLayer;
import neural_network.layers.layer.NeuralLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizer.Optimizer;
import nnarray.NNArray;
import nnarray.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class DeepNeuralNetwork {
    private ArrayList<DenseNeuralLayer> layers;

    private int inputSize;
    private boolean trainable;

    private FunctionLoss functionLoss;
    private Optimizer optimizer;

    public DeepNeuralNetwork() {
        layers = new ArrayList<>();
        trainable = true;
    }

    public DeepNeuralNetwork addInputLayer(int size) {
        inputSize = size;

        return this;
    }

    public DeepNeuralNetwork create() {
        layers.get(0).initialize(inputSize);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).initialize(layers.get(i - 1).getCountNeuron());
        }

        for (DenseNeuralLayer layer : layers) {
            layer.initialize(optimizer);
        }

        return this;
    }

    public DeepNeuralNetwork setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public DeepNeuralNetwork addLayer(DenseNeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public DeepNeuralNetwork setFunctionLoss(FunctionLoss functionLoss) {
        this.functionLoss = functionLoss;

        return this;
    }

    public void info() {
        System.out.println("\t\t\t\t DEEP NEURAL NETWORK ");
        System.out.println("==========================================================");
        System.out.println("Layer\t\t| Input size \t| Output size \t| Count param");
        for (DenseNeuralLayer neuralLayer : layers) {
            System.out.println("____________|_______________|_______________|_____________");
            neuralLayer.info();
        }
        System.out.println("____________|_______________|_______________|_____________");
    }

    public DeepNeuralNetwork setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;

        return this;
    }

    public void save(FileWriter fileWriter) throws IOException {
        fileWriter.write("Deep neural network\n");
        fileWriter.write(inputSize + "\n");
        for (DenseNeuralLayer layer : layers) {
            layer.write(fileWriter);
        }
        fileWriter.write("End\n");
        fileWriter.flush();
    }

    public static DeepNeuralNetwork read(Scanner scanner) throws Exception {
        if (scanner.nextLine().equals("Deep neural network")) {
            DeepNeuralNetwork network = new DeepNeuralNetwork()
                    .addInputLayer(Integer.parseInt(scanner.nextLine()));
            DenseNeuralLayer.read(scanner, network.layers);
            network.create();
            return network;
        }
        throw new Exception("Network is not deep");
    }

    private void queryTrain(NNMatrix input) {
        layers.get(0).generateTrainOutput(input);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).generateTrainOutput(layers.get(i - 1).getOutputs());
        }
    }

    public float train(NNMatrix input, NNMatrix idealOutput) {
        queryTrain(input);
        backpropagation(findDerivative(idealOutput));
        update();
        return functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutputs(), idealOutput);
    }

    private NNMatrix findDerivative(NNMatrix idealOutput) {
        NNMatrix error = new NNMatrix(idealOutput);
        functionLoss.findDerivative(layers.get(layers.size() - 1).getOutputs(), idealOutput, error);

        return error;
    }

    public void update() {
        optimizer.update();
        for (NeuralLayer layer : layers) {
            layer.update(optimizer);
        }
    }

    private void backpropagation(NNMatrix error) {
        layers.get(layers.size() - 1).generateError(error);
        if (trainable) {
            layers.get(layers.size() - 1).generateErrorWeight(error);
        }
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers.get(i).generateError(layers.get(i + 1).getError());
            if (trainable) {
                layers.get(i).generateErrorWeight(layers.get(i + 1).getError());
            }
        }
    }

    public float accuracy(NNMatrix input, NNMatrix idealOutput) {
        queryTrain(input);
        NNArray error = new NNMatrix(idealOutput);
        return functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutputs(), idealOutput);
    }
}
