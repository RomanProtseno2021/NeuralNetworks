package data.mnist;

import lombok.Getter;
import nnarray.NNMatrix;
import nnarray.NNTensor4D;
import nnarray.NNVector;

import javax.annotation.processing.Generated;

public class Image1dData {
    @Getter
    private NNVector inputs;
    @Getter
    private NNVector outputs;

    public Image1dData(NNVector inputs, NNVector outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}
