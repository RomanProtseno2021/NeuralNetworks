package neural_network.layers.layer;

import neural_network.optimizer.Optimizer;
import nnarray.NNArray;
import nnarray.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;

public abstract class NeuralLayer {
    public abstract void initialize(Optimizer optimizer);

    public abstract void update(Optimizer optimizer);

    public abstract void info();

    public abstract void write(FileWriter writer) throws IOException;
}
