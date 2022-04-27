package neural_network.layers.dense;

import neural_network.activation.FunctionActivation;
import neural_network.optimizer.Optimizer;
import nnarray.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class DropoutLayer extends DenseNeuralLayer {
    private final double dropout;

    public DropoutLayer(double dropout) {
        this.dropout = dropout;
    }

    @Override
    public void initialize(int size) {
        this.countNeuron = size;
    }

    @Override
    public void generateOutput(NNMatrix input) {
        this.outputs = input;
    }

    @Override
    public void generateTrainOutput(NNMatrix input) {
        outputs = new NNMatrix(input);
        outputs.dropout(input, dropout);
    }

    @Override
    public void generateError(NNMatrix error) {
        this.error = new NNMatrix(error);
        this.error.dropoutBack(error, dropout);
    }

    @Override
    public void generateErrorWeight(NNMatrix error) {
        //no have betta
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have betta
    }

    @Override
    public void update(Optimizer optimizer) {
        //no have betta
    }

    @Override
    public void info() {
        System.out.println("Dropout \t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|");
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Dropout layer\n");
        writer.write(dropout + "\n");
        writer.flush();
    }

    public static DropoutLayer read(Scanner scanner) {
        return new DropoutLayer(Double.parseDouble(scanner.nextLine()));
    }
}
