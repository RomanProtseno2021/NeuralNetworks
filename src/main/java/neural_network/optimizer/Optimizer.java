package neural_network.optimizer;

import lombok.Data;
import lombok.Getter;
import nnarray.NNArray;

@Data
public abstract class Optimizer {
    protected float clipValue = 0;
    @Getter
    protected int countParam = 0;

    protected int t = 0;

    public void update() {
        t++;
    }

    public abstract void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam);

    public void setClipValue(double clipValue) {
        this.clipValue = (float) clipValue;
    }

    public float getClipValue() {
        return clipValue;
    }
}
