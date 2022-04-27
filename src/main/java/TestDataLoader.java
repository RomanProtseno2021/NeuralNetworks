import data.mnist.BatchMNIST;
import data.mnist.MNIST1dLoader;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.DeepNeuralNetwork;
import neural_network.optimizer.AdamOptimizer;
import neural_network.optimizer.SGDOptimizer;
import nnarray.NNMatrix;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;

public class TestDataLoader {
    public static void main(String[] args) throws IOException {
        DeepNeuralNetwork network = new DeepNeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(64)
                        .setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new BatchNormalizationLayer())
                .addLayer(new DenseLayer(64)
                        .setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new BatchNormalizationLayer())
                .addLayer(new DenseLayer(64)
                        .setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new BatchNormalizationLayer())
                .addLayer(new DenseLayer(10))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();

        MNIST1dLoader loader = new MNIST1dLoader(BatchMNIST.MNIST);

        DataTrainer trainer = new DataTrainer(60000, 10000, loader);

        for (int i = 0; i < 10; i++) {
            trainer.train(network, 128, 10);
            network.save(new FileWriter("test_mnist.txt"));
        }
    }
}
