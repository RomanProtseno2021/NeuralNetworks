package neural_network.activation;

import neural_network.regularization.Regularization;
import nnarray.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public interface FunctionActivation {
    void activation(NNArray input, NNArray output);

    void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta);

    void save(FileWriter writer) throws IOException;

    static FunctionActivation read(Scanner scanner){
        FunctionActivation functionActivation;

        String activ = scanner.nextLine();
        functionActivation = switch (activ) {
            case "ReLU" -> new FunctionActivation.ReLU();
            case "LeakyReLU" -> new FunctionActivation.LeakyReLU(Double.parseDouble(scanner.nextLine()));
            case "Sigmoid" -> new FunctionActivation.Sigmoid();
            case "Tanh" -> new FunctionActivation.Tanh();
            default -> null;
        };

        return functionActivation;
    }

    class ReLU implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.relu(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derRelu(input, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("ReLU\n");
        }
    }

    class Sigmoid implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.sigmoid(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derSigmoid(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Sigmoid\n");
        }
    }

    class Tanh implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.tanh(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derTanh(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Tanh\n");
        }
    }

    class LeakyReLU implements FunctionActivation {
        private final float param;

        public LeakyReLU(double param) {
            this.param = (float) param;
        }

        public LeakyReLU() {
            this.param = 0.01f;
        }

        @Override
        public void activation(NNArray input, NNArray output) {
            output.leakyRelu(input, param);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derLeakyRelu(input, error, param);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("LeakyReLU\n");
            writer.write(param + "\n");
        }
    }
}
