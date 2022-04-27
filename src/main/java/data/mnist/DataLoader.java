package data.mnist;

import nnarray.NNMatrix;

import java.util.Collections;

public abstract class DataLoader {
    public abstract NNMatrix[] getNextTrainData(int sizeBatch);

    public abstract NNMatrix[] getNextTestData(int sizeBatch);
}
