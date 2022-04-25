package nnarray;

import lombok.Getter;

public class NNTensor4D extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private final int length;

    private final int[] dVal;
    private final int[] lVal;
    private final int[] iVal;

    public NNTensor4D(int depth, int length, int row, int column) {
        super(column * row * depth * length);
        this.column = column;
        this.row = row;
        this.depth = depth;
        this.length = length;

        dVal = new int[depth];
        lVal = new int[length];
        iVal = new int[row];
        int sq = column * row * length;
        int sql = column * row;
        for (int i = 0; i < depth; i++) {
            dVal[i] = i * sq;
        }
        for (int i = 0; i < length; i++) {
            lVal[i] = i * sql;
        }
        for (int i = 0; i < row; i++) {
            iVal[i] = i * row;
        }
    }

    public NNTensor4D(NNTensor4D tensor) {
        this(tensor.depth,tensor.length, tensor.row, tensor.column);
    }

    public float get(int i, int j, int k, int l) {
        return data[dVal[i] + lVal[j] + iVal[k] + l];
    }

    public void set(int i, int j, int k, int l, float value) {
        data[dVal[i] + lVal[j] + iVal[k] + l] = value;
    }
}
