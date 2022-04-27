package data.mnist;

import nnarray.NNMatrix;
import nnarray.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class MNIST1dLoader extends DataLoader {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[784];
    private byte[] byteNumb = new byte[1];

    private ArrayList<Image1dData> train;
    private ArrayList<Image1dData> test;

    private int curTrain = 0;
    private int curTest = 0;

    private static FileInputStream scanner;
    private static FileInputStream scannerNumb;
    private static FileInputStream scannerTest;
    private static FileInputStream scannerNumbTest;

    private BatchMNIST batchMNIST;

    private void loadTrainFilesWithNumber() {
        try {
            scanner = new FileInputStream("D:/datasets/mnist_batch/" + batchMNIST.getTrainFile());
            scannerNumb = new FileInputStream("D:/datasets/mnist_batch/" + batchMNIST.getTrainFileMark());

            scanner.skip(16);
            scannerNumb.skip(8);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTestFilesWithNumber() {
        try {
            scannerTest = new FileInputStream("D:/datasets/mnist_batch/" + batchMNIST.getTestFile());
            scannerNumbTest = new FileInputStream("D:/datasets/mnist_batch/" + batchMNIST.getTestFileMark());

            scannerTest.skip(16);
            scannerNumbTest.skip(8);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public MNIST1dLoader() {
        this(BatchMNIST.MNIST);
    }

    public MNIST1dLoader(BatchMNIST batchMNIST) {
        this(batchMNIST, new TransformData.Sigmoid());
    }

    public MNIST1dLoader(BatchMNIST batchMNIST, TransformData transform) {
        this.batchMNIST = batchMNIST;
        this.transformData = transform;

        loadTrainFilesWithNumber();
        loadTestFilesWithNumber();

        train = new ArrayList<>(batchMNIST.getSizeTrain());
        test = new ArrayList<>(batchMNIST.getSizeTest());

        try {
            loadTrainData();
            loadTestData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
        for (int i = 0; i < batchMNIST.getSizeTrain(); i++) {
            if (scannerNumb.available() > 0) {
                scannerNumb.read(byteNumb);
                scanner.read(bytes);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(batchMNIST.getCountClass());
                output.set(trueNumb, 1);
                train.add(new Image1dData(new NNVector(inputsData), output));
            }
        }
        Collections.shuffle(train);
    }

    private void loadTestData() throws IOException {
        for (int i = 0; i < batchMNIST.getSizeTrain(); i++) {
            if (scannerNumbTest.available() > 0) {
                scannerNumbTest.read(byteNumb);
                scannerTest.read(bytes);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(batchMNIST.getCountClass());
                output.set(trueNumb, 1);
                test.add(new Image1dData(new NNVector(inputsData), output));
            }
        }
        Collections.shuffle(test);
    }

    public NNMatrix[] getNextTrainData(int sizeBatch) {
        if (curTrain + sizeBatch < batchMNIST.getSizeTrain()) {
            NNMatrix[] result = new NNMatrix[2];
            result[0] = new NNMatrix(sizeBatch, 784);
            result[1] = new NNMatrix(sizeBatch, batchMNIST.getCountClass());
            for (int i = 0; i < sizeBatch; i++) {
                result[0].setVector(train.get(curTrain).getInputs(), i);
                result[1].setVector(train.get(curTrain).getOutputs(), i);
                curTrain++;
            }
            return result;
        } else {
            int size = batchMNIST.getSizeTrain() - curTrain;
            NNMatrix[] result = new NNMatrix[2];
            result[0] = new NNMatrix(size, 784);
            result[1] = new NNMatrix(size, batchMNIST.getCountClass());
            for (int i = 0; i < size; i++) {
                result[0].setVector(train.get(curTrain).getInputs(), i);
                result[1].setVector(train.get(curTrain).getOutputs(), i);
                curTrain++;
            }
            curTrain = 0;
            Collections.shuffle(train);
            return result;
        }
    }

    public NNMatrix[] getNextTestData(int sizeBatch) {
        NNMatrix[] result = new NNMatrix[2];
        if (curTest + sizeBatch < batchMNIST.getSizeTest()) {
            result[0] = new NNMatrix(sizeBatch, 784);
            result[1] = new NNMatrix(sizeBatch, batchMNIST.getCountClass());
            for (int i = 0; i < sizeBatch; i++) {
                result[0].setVector(test.get(curTest).getInputs(), i);
                result[1].setVector(test.get(curTest).getOutputs(), i);
                curTest++;
            }

            return result;
        } else {
            int size = batchMNIST.getSizeTest() - curTest;
            result[0] = new NNMatrix(size, 784);
            result[1] = new NNMatrix(size, batchMNIST.getCountClass());
            for (int i = 0; i < size; i++) {
                result[0].setVector(test.get(curTest).getInputs(), i);
                result[1].setVector(test.get(curTest).getOutputs(), i);
                curTest++;
            }
            curTest = 0;
            Collections.shuffle(train);
            return result;
        }
    }

    private void generateInput() {
        inputsData = new float[784];
        for (int i = 0; i < 784; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
    }
}