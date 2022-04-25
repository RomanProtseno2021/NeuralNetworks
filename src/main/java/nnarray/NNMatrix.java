package nnarray;

import lombok.Getter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Vector;

public class NNMatrix extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;

    private final int[] iVal;

    public NNMatrix(int row, int column) {
        super(column * row);
        this.column = column;
        this.row = row;

        iVal = new int[row];
        for (int i = 0; i < row; i++) {
            iVal[i] = i * column;
        }
    }

    public NNMatrix(NNMatrix matrix) {
        this(matrix.row, matrix.column);
    }

    public float get(int i, int j) {
        return data[iVal[i] + j];
    }

    public void set(int i, int j, float value) {
        data[iVal[i] + j] = value;
    }

    public NNVector sumRow() {
        NNVector result = new NNVector(column);
        int i0;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            for (int j = 0; j < column; j++) {
                result.data[j] += data[i0 + j];
            }
        }
        return result;
    }

    public NNVector mulVector(NNVector vector) {
        NNVector nnVector = new NNVector(row);
        if (column == vector.size) {
            int value;
            int i0;
            for (int i = 0; i < row; i++) {
                value = 0;
                i0 = i * column;
                for (int j = 0; j < column; j++) {
                    value += data[i0 + j] * vector.data[j];
                }
                nnVector.set(i, value);
            }
        } else {
            throw new ArithmeticException("Don't can mul matrix");
        }

        return nnVector;
    }

    public NNMatrix mul(NNMatrix matrix) {
        NNMatrix result = new NNMatrix(row, matrix.column);
        NNMatrix matrixT = matrix.transpose();
        int i0, i1, j0;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            i1 = result.iVal[i];
            for (int j = 0; j < result.column; j++) {
                j0 = matrixT.iVal[j];
                for (int k = 0; k < column; k++) {
                    result.data[i1 + j] += data[i0 + k] * matrixT.data[j0 + k];
                }
            }
        }
        return result;
    }

    public NNMatrix mulT(NNMatrix matrix) {
        NNMatrix result = new NNMatrix(row, matrix.row);
        int i0, i1, j0;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            i1 = result.iVal[i];
            for (int j = 0; j < result.column; j++) {
                j0 = matrix.iVal[j];
                for (int k = 0; k < column; k++) {
                    result.data[i1 + j] += data[i0 + k] * matrix.data[j0 + k];
                }
            }
        }
        return result;
    }

    public void setVector(NNVector vector, int index) {
        int s = iVal[index];
        System.arraycopy(vector.data, 0, data, s, vector.size);
    }

    public void addVector(NNVector vector) {
        int i0;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            for (int j = 0; j < vector.size; j++) {
                data[i0 + j] += vector.data[j];
            }
        }
    }

    public NNMatrix transpose() {
        NNMatrix nnMatrix = new NNMatrix(this.column, this.row);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                nnMatrix.data[i + j * row] = data[iVal[i] + j];
            }
        }
        return nnMatrix;
    }

    public NNVector mulTransposeVector(NNVector vector) {
        NNVector nnVector = new NNVector(column);
        if (row == vector.size) {
            int value;
            for (int i = 0; i < column; i++) {
                value = 0;
                for (int j = 0; j < row; j++) {
                    value += data[iVal[j] + i] * vector.data[j];
                }
                nnVector.set(i, value);
            }
        } else {
            throw new ArithmeticException("Don't can mul matrix");
        }

        return nnVector;
    }

    public void addMulVector(NNVector first, NNVector second) {
        for (int i = 0; i < second.getLength(); i++) {
            for (int j = 0; j < first.getLength(); j++) {
                data[iVal[i] + j] += second.getData()[i] * first.getData()[j];
            }
        }
    }

    public void save(FileWriter writer) throws IOException {
        writer.write(row + " " + column + "\n");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                writer.write(data[iVal[i] + j] + " ");
                if (j % 1000 == 0) {
                    writer.flush();
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }

    public static NNMatrix read(Scanner scanner){
        int[] size  = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        NNMatrix matrix = new NNMatrix(size[0], size[1]);
        for (int i = 0; i < matrix.row; i++) {
            double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
            for (int j = 0; j < matrix.column; j++) {
                matrix.data[matrix.iVal[i] + j] = (float) arr[j];
            }
        }
        return matrix;
    }
}
