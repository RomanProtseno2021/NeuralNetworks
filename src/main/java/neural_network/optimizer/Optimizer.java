package neural_network.optimizer;

import lombok.Data;
import lombok.Getter;
import nnarray.NNArray;

@Data
public abstract class Optimizer {
    protected float clipValue = 0;
    protected boolean setClipValue = false;
    @Getter
    protected int countParam = 0;

    public void update() {
        //no action
    }

    public abstract void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam);

    public void setClipValue(double clipValue) {
        this.clipValue = (float) clipValue;
        setClipValue = true;
    }

    public float getClipValue() {
        return clipValue;
    }
}
