package neural_network.layers.dense;

import neural_network.activation.FunctionActivation;
import neural_network.optimizer.Optimizer;
import nnarray.NNMatrix;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ActivationLayer extends DenseNeuralLayer {
    private final FunctionActivation functionActivation;
    private NNMatrix input;

    public ActivationLayer(FunctionActivation functionActivation) {
        this.functionActivation = functionActivation;
    }

    @Override
    public void initialize(int size) {
        this.countNeuron = size;
    }

    @Override
    public void generateOutput(NNMatrix input) {
        this.input = input;
        outputs = new NNMatrix(input);
        functionActivation.activation(input, outputs);
    }

    @Override
    public void generateTrainOutput(NNMatrix input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNMatrix error) {
        this.error = new NNMatrix(error);
        functionActivation.derivativeActivation(input, outputs, error, this.error);
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
        System.out.println("Activation\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|");
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Activation layer\n");
        functionActivation.save(writer);
        writer.flush();
    }

    public static ActivationLayer read(Scanner scanner) {
        return new ActivationLayer(FunctionActivation.read(scanner));
    }
}
