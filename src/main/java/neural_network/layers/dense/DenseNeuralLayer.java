package neural_network.layers.dense;

import lombok.Getter;
import lombok.Setter;
import neural_network.layers.layer.NeuralLayer;
import neural_network.regularization.Regularization;
import nnarray.NNMatrix;

import java.util.ArrayList;
import java.util.Scanner;

public abstract class DenseNeuralLayer extends NeuralLayer{
    @Setter
    @Getter
    protected NNMatrix outputs;
    @Setter
    @Getter
    protected NNMatrix error;
    @Getter
    protected int countNeuron;

    public abstract void initialize(int size);

    public abstract void generateOutput(NNMatrix input);

    public abstract void generateTrainOutput(NNMatrix input);

    public abstract void generateError(NNMatrix error);

    public abstract void generateErrorWeight(NNMatrix error);

    public static void read(Scanner scanner, ArrayList<DenseNeuralLayer> layers){
        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            switch (layer) {
                case "Dense layer" -> layers.add(DenseLayer.read(scanner));
                case "Dropout layer" -> layers.add(DropoutLayer.read(scanner));
                case "Activation layer" -> layers.add(ActivationLayer.read(scanner));
                case "Batch normalization layer" -> layers.add(BatchNormalizationLayer.read(scanner));
            }

            layer = scanner.nextLine();
        }
    }
}
