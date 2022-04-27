package neural_network.layers.convolution;

import lombok.Getter;
import lombok.Setter;
import neural_network.layers.dense.*;
import neural_network.layers.layer.NeuralLayer;
import nnarray.NNMatrix;
import nnarray.NNTensor;

import java.util.ArrayList;
import java.util.Scanner;

public abstract class Convolution1DNeuralLayer extends NeuralLayer {
    @Setter
    @Getter
    protected NNTensor outputs;
    @Setter
    @Getter
    protected NNTensor error;
    @Getter
    protected int width, depth;

    public abstract void initialize(int width, int depth);

    public abstract void generateOutput(NNTensor input);

    public abstract void generateTrainOutput(NNTensor input);

    public abstract void generateError(NNTensor error);

    public abstract void generateErrorWeight(NNTensor error);

    public static void read(Scanner scanner, ArrayList<Convolution1DNeuralLayer> layers){
        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            switch (layer) {
//                case "Dense layer" -> layers.add(DenseLayer.read(scanner));
//                case "Dropout layer" -> layers.add(DropoutLayer.read(scanner));
//                case "Activation layer" -> layers.add(ActivationLayer.read(scanner));
//                case "Batch normalization layer" -> layers.add(BatchNormalizationLayer.read(scanner));
            }

            layer = scanner.nextLine();
        }
    }
}
