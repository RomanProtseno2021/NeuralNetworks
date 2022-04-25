import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.DeepNeuralNetwork;
import neural_network.optimizer.*;
import nnarray.NNMatrix;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class Test {
    public static void main(String[] args) throws Exception {
        NNMatrix input = new NNMatrix(100, 784);
        new Initializer.RandomNormal().initialize(input);
//        input.save(new FileWriter("opt_input.txt"));
        NNMatrix output = new NNMatrix(100, 1);
        new Initializer.RandomNormal().initialize(output);
        output.sigmoid(output);
//        output.save(new FileWriter("opt_output.txt"));
//
        DeepNeuralNetwork network = new DeepNeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(128))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
//                .addLayer(new DropoutLayer(0.25))
                .addLayer(new DenseLayer(64))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
//                .addLayer(new DropoutLayer(0.25))
                .addLayer(new DenseLayer(1))
                .addLayer(new ActivationLayer(new FunctionActivation.Sigmoid()))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();
//
//        network.save(new FileWriter("optimize_network.txt"));

//        NNMatrix input = NNMatrix.read(new Scanner(new File("opt_input.txt")));
//        NNMatrix output = NNMatrix.read(new Scanner(new File("opt_output.txt")));
//        DeepNeuralNetwork network = DeepNeuralNetwork.read(new Scanner(new File("optimize_network.txt")))
//                .setFunctionLoss(new FunctionLoss.Quadratic())
//                .setOptimizer(new AdamOptimizer());

        network.info();

        for (int i = 0; i < 1000; i++) {
            System.out.println(network.train(input, output));
        }
    }
}
