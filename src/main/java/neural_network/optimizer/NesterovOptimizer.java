package neural_network.optimizer;

import lombok.Data;
import nnarray.NNArray;

@Data
public class NesterovOptimizer extends Optimizer {
    /**
     * rt - retention rate
     * Nesterov
     * θ(t) = rt * θ(t-1) - lr * dw(t)
     * w(t) = w(t-1) + rt * θ(t) - lr * dw(t)
     */
    private final float learningRate;
    private final float retentionRate;

    public NesterovOptimizer(double learningRate, double retentionRate) {
        this.learningRate = (float) learningRate;
        this.retentionRate = (float) retentionRate;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentumN(deltaWeight, retentionRate, learningRate);
        weight.addMomentumN(deltaWeight, additionParam[0], retentionRate, learningRate);
        deltaWeight.clear();
    }
}
