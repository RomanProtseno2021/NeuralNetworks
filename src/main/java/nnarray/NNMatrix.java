package nnarray;

import lombok.Getter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

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
            for (int j = 0; j < column; j++, i0++) {
                result.data[j] += data[i0];
            }
        }
        return result;
    }

    public NNVector mulAndSumRow(NNMatrix mulMatrix) {
        NNVector result = new NNVector(column);
        int i0;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            for (int j = 0; j < column; j++, i0++) {
                result.data[j] += data[i0] * mulMatrix.data[i0];
            }
        }
        return result;
    }

    public NNVector sumRowSubPow2(NNVector vector) {
        NNVector result = new NNVector(column);
        int i0;
        float sub;
        for (int i = 0; i < row; i++) {
            i0 = iVal[i];
            for (int j = 0; j < column; j++, i0++) {
                sub = data[i0] - vector.data[j];
                result.data[j] += sub * sub;
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

    public NNMatrix normalization(NNVector mean, NNVector var) {
        NNMatrix result = new NNMatrix(this);
        int index;
        float[] varSqrt = new float[var.size];
        for (int i = 0; i < var.size; i++) {
            varSqrt[i] = (float) (Math.sqrt(var.data[i]) + 0.0000001f);
        }
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[index] = (data[index] - mean.data[j]) / (varSqrt[j]);
            }
        }
        return result;
    }

    public NNMatrix mulAndAdd(NNVector mulVec, NNVector addVec) {
        NNMatrix result = new NNMatrix(this);
        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[index] = data[index] * mulVec.data[j] + addVec.data[j];
            }
        }
        return result;
    }

    public NNVector derVar(NNMatrix input, NNVector mean, NNVector var, NNVector gamma) {
        NNVector result = new NNVector(mean);
        float[] dVar = new float[var.size];
        for (int i = 0; i < var.size; i++) {
            dVar[i] = (float) (-0.5 * gamma.data[i] * Math.pow(var.data[i] + 0.00000001f, -1.5));
        }

        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[j] += data[index] * (input.data[index] - mean.data[j]);
            }
        }
        for (int i = 0; i < column; i++) {
            result.data[i] *= dVar[i];
        }
        return result;
    }

    public NNVector derMean(NNMatrix input, NNVector mean, NNVector var, NNVector derVar, NNVector gamma) {
        NNVector result = new NNVector(mean.size);
        float[] dMean = new float[mean.size];
        float[] dVar = new float[var.size];
        for (int i = 0; i < var.size; i++) {
            dMean[i] = (float) (-gamma.data[i] / Math.sqrt(var.data[i] + 0.00000001f));
        }

        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[j] += data[index];
                dVar[j] = input.data[index] - mean.data[j];
            }
        }
        for (int i = 0; i < column; i++) {
            result.data[i] *= dMean[i];
            result.data[i] += -2 * derVar.data[i] * dVar[i] / row;
        }
        return result;
    }

    public NNMatrix derNorm(NNMatrix input, NNVector mean, NNVector derMean, NNVector var, NNVector derVar) {
        NNMatrix result = new NNMatrix(input);
        derMean.div(row);
        derVar.mul(2.0f / row);

        float[] dVar = new float[var.size];
        for (int i = 0; i < var.size; i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var.data[i] + 0.00000001f));
        }

        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[index] = data[index] * dVar[j] + derVar.data[j] * (input.data[index] - mean.data[j]) + derMean.data[j];
            }
        }

        return result;
    }

    public NNMatrix mul(NNVector mulVec) {
        NNMatrix result = new NNMatrix(this);
        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                result.data[index] = data[index] * mulVec.data[j];
            }
        }
        return result;
    }

    public NNMatrix mul(NNMatrix matrix) {
        NNMatrix result = new NNMatrix(row, matrix.column);
        NNMatrix matrixT = matrix.transpose();
        int i0, i1, j0;
        float sum;
        for (int i = 0; i < row; i++) {
            i1 = result.iVal[i];
            for (int j = 0; j < result.column; j++, i1++) {
                i0 = iVal[i];
                j0 = matrixT.iVal[j];
                sum = 0;
                for (int k = 0; k < column; k++, i0++, j0++) {
                    sum += data[i0] * matrixT.data[j0];
                }
                result.data[i1] = sum;
            }
        }
        return result;
    }

    public NNMatrix mulT(NNMatrix matrix) {
        NNMatrix result = new NNMatrix(row, matrix.row);
        int i0, i1, j0;
        float sum;
        for (int i = 0; i < row; i++) {
            i1 = result.iVal[i];
            for (int j = 0; j < result.column; j++, i1++) {
                i0 = iVal[i];
                j0 = matrix.iVal[j];
                sum = 0;
                for (int k = 0; k < column; k++, i0++, j0++) {
                    sum += data[i0] * matrix.data[j0];
                }
                result.data[i1] = sum;
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
        int index;
        for (int i = 0; i < row; i++) {
            index = iVal[i];
            for (int j = 0; j < column; j++, index++) {
                nnMatrix.data[i + nnMatrix.iVal[j]] = data[index];
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

    public static NNMatrix read(Scanner scanner) {
        int[] size = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
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
