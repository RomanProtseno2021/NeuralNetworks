package nnarray;

import lombok.Getter;

import static java.lang.Math.log;
import static java.lang.Math.pow;

public class NNArray {
    @Getter
    protected float[] data;
    @Getter
    protected final int size;

    public NNArray(int size) {
        this.size = size;
        this.data = new float[size];
    }

    public NNArray add(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] += vector.data[i];
        }
        return this;
    }

    public float get(int i) {
        return data[i];
    }

    public void set(int i, float value) {
        data[i] = value;
    }

    public void sub(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] -= vector.data[i];
        }
    }

    public void clip(float val) {
        float a;
        final float valMin = -val;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > val) {
                data[i] = val;
            } else if (a < valMin) {
                data[i] = valMin;
            }
        }
    }

    public void subDivSqrt(NNArray nominator, NNArray denominator, float lr) {
        for (int i = 0; i < size; i++) {
            data[i] -= lr * nominator.data[i] / (Math.sqrt(denominator.data[i]) + 0.0000001f);
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        for (int i = 0; i < size; i++) {
            data[i] -= lr * (nominator.data[i] / (normN + 0.0000001f)) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public NNArray divSqrt(NNArray nominator, NNArray denominator) {
        NNArray result = new NNArray(nominator.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) (data[i] * Math.sqrt(nominator.data[i] + 0.0000001f) / (Math.sqrt(denominator.data[i]) + 0.0000001f));
        }
        return result;
    }

    public void subDivSqrt(NNArray nominator, NNArray denominator) {
        for (int i = 0; i < size; i++) {
            data[i] -= nominator.data[i] / (Math.sqrt(denominator.data[i]) + 0.0000001f);
        }
    }

    public void addPow2(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] += vector.data[i] * vector.data[i];
        }
    }

    public void momentumPow2(NNArray vector, final float decay) {
        final float dr = 1 - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + dr * vector.data[i] * vector.data[i];
        }
    }

    public void addAndMul(NNArray vector, float val) {
        for (int i = 0; i < size; i++) {
            data[i] += val * vector.data[i];
        }
    }

    public void subAndMul(NNArray vector, float val) {
        for (int i = 0; i < size; i++) {
            data[i] -= val * vector.data[i];
        }
    }

    public void momentum(NNArray array, final float decay) {
        final float rt = 1.0f - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + array.data[i] * rt;
        }
    }

    public void momentumM(NNArray array, final float decay) {
        final float rt = 1.0f - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] - array.data[i] * rt;
        }
    }

    public void dropout(NNArray input, double chanceDrop) {
        float drop = (float) (1.0f / (1.0f - chanceDrop));
        for (int i = 0; i < size; i++) {
            if (Math.random() > chanceDrop) {
                data[i] = input.data[i] * drop;
            }
        }
    }

    public void dropoutBack(NNArray input, double chanceDrop) {
        float drop = (float) (1.0f / (1.0f - chanceDrop));
        for (int i = 0; i < size; i++) {
            if (input.data[i] != 0) {
                data[i] = input.data[i] * drop;
            }
        }
    }

    public void momentumN(NNArray array, final float decay, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] - array.data[i] * lr;
        }
    }

    public void addMomentumN(NNArray derivative, NNArray decay, final float decayR, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] += decayR * decay.data[i] - derivative.data[i] * lr;
        }
    }

    public void sub(NNArray vector1, NNArray vector2) {
        for (int i = 0; i < size; i++) {
            data[i] = vector1.data[i] - vector2.data[i];
        }
    }

    public float subPow2(NNArray vector) {
        float sum = 0;
        float difference;
        for (int i = 0; i < size; i++) {
            difference = data[i] - vector.data[i];
            sum += difference * difference;
        }
        return sum;
    }

    public void mul(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] *= vector.data[i];
        }
    }

    public void div(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] /= vector.data[i];
        }
    }

    public void div(float val) {
        for (int i = 0; i < size; i++) {
            data[i] /= val;
        }
    }

    public void mul(float val) {
        for (int i = 0; i < size; i++) {
            data[i] *= val;
        }
    }

    public void clear() {
        for (int i = 0; i < size; i++) {
            data[i] = 0;
        }
    }

    public void sub(float val) {
        for (int i = 0; i < size; i++) {
            data[i] -= val;
        }
    }

    public void subSign(float val) {
        float a;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > 0) {
                data[i] -= val;
            } else if (a < 0) {
                data[i] += val;
            }
        }
    }

    public void fill(float value) {
        for (int i = 0; i < size; i++) {
            data[i] = value;
        }
    }

    public void fill(int index, float value) {
        data[index] = value;
    }

    public void binaryCrossEntropy(NNArray outputs, NNArray idealOutputs) {
        for (int i = 0; i < size; i++) {
            data[i] = (outputs.data[i] - idealOutputs.data[i]) / ((1 - outputs.data[i]) * outputs.data[i] + 0.00000001f);
        }
    }

    public void crossEntropy(NNArray outputs, NNArray idealOutputs) {
        for (int i = 0; i < size; i++) {
            data[i] = -idealOutputs.data[i] / (outputs.data[i] + 0.0000001f);
        }
    }

    public void relu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, input.data[i]);
        }
    }

    public void silu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (data[i] / (1 + pow(Math.E, -data[i])));
        }
    }

    public void derRelu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            }
        }
    }

    public void derSilu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (error.data[i] * ((1 + pow(Math.E, -input.data[i]) + input.data[i] * pow(Math.E, -input.data[i]))
                    / Math.pow(1 + pow(Math.E, -input.data[i]), 2)));
        }
    }

    public void derSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = output.data[i] * (1 - output.data[i]) * error.data[i];
        }
    }

    public void derTanh(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (1 - output.data[i] * output.data[i]) * error.data[i];
        }
    }

    public void derLeakyRelu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = param * error.data[i];
            }
        }
    }

    public void derElu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = (float) (param * Math.pow(Math.E, input.data[i]) * error.data[i]);
            }
        }
    }

    public void sigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (1.0 / (1 + Math.pow(Math.E, -input.data[i])));
        }
    }

    public void gaussian(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.pow(Math.E, -input.data[i] * input.data[i]));
        }
    }

    public void derGaussian(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (-2 * input.data[i] * Math.pow(Math.E, -input.data[i] * input.data[i]) * error.data[i]);
        }
    }

    public float max() {
        float max = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] > max) {
                max = data[i];
            }
        }
        return max;
    }

    public void softmax(NNArray input) {
        if (input instanceof NNMatrix) {
            int index;
            final int rows = ((NNMatrix) input).getRow();
            final int column = ((NNMatrix) input).getColumn();
            for (int k = 0; k < rows; k++) {
                float sum = 0;
                index = k * column;
                float max = input.data[index];
                for (int i = 1; i < column; i++, index++) {
                    if (max < data[index])
                        max = data[index];
                }
                index = k * column;
                for (int i = 0; i < column; i++, index++) {
                    data[index] = (float) (Math.pow(Math.E, input.data[index] - max));
                    sum += data[index];
                }

                index = k * column;
                for (int i = 0; i < column; i++, index++) {
                    data[index] /= sum;
                }
            }
        }
    }

    public void derSoftmax(NNArray output, NNArray error) {
        if (output instanceof NNMatrix) {
            int index, indexI, indexJ;
            final int rows = ((NNMatrix) output).getRow();
            final int column = ((NNMatrix) output).getColumn();
            for (int k = 0; k < rows; k++) {
                float value;
                index = k * column;
                indexI = index;
                for (int i = 0; i < column; i++, indexI++) {
                    data[indexI] = 0;
                    indexJ = index;
                    for (int j = 0; j < column; j++, indexJ++) {
                        if (i != j) {
                            value = output.data[indexI] * -output.data[indexJ];
                        } else {
                            value = output.data[indexI] * (1 - output.data[indexI]);
                        }
                        data[indexI] += error.getData()[indexJ] * value;
                    }
                }
            }
        }
    }

    public void tanh(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.tanh(input.data[i]);
        }
    }

    public void linear(NNArray input) {
        System.arraycopy(input.data, 0, data, 0, size);
    }

    public void atan(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.atan(input.data[i]);
        }
    }

    public void elu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = (float) ((Math.pow(Math.E, input.data[i]) - 1) * param);
            }
        }
    }

    public void softplus(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.log(Math.pow(Math.E, input.data[i]) + 1);
        }
    }

    public void hardSigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, Math.min(1, input.data[i] * 0.2f + 0.5f));
        }
    }

    public void derHardSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (output.data[i] >= 0 && output.data[i] <= 1) {
                data[i] = 0.2f * error.data[i];
            }
        }
    }

    public void leakyRelu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = input.data[i] * param;
            }
        }
    }
}
