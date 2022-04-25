package nnarray;

import lombok.Getter;

public class NNTensor extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;

    private final int[] dVal;
    private final int[] iVal;

    public NNTensor(int depth, int row, int column) {
        super(column * row * depth);
        this.column = column;
        this.row = row;
        this.depth = depth;

        dVal = new int[depth];
        iVal = new int[row];
        int sq = column * row;
        for (int i = 0; i < depth; i++) {
            dVal[i] = i * sq;
        }
        for (int i = 0; i < row; i++) {
            iVal[i] = i * row;
        }
    }

    public NNTensor(NNTensor tensor) {
        this(tensor.depth, tensor.row, tensor.column);
    }

    public float get(int i, int j, int k) {
        return data[dVal[i] + iVal[j] + k];
    }

    public void set(int i, int j, int k, float value) {
        data[dVal[i] + iVal[j] + k] = value;
    }
}
