package neural_network.loss;

import lombok.NoArgsConstructor;
import nnarray.NNArray;

import static java.lang.Math.log;

public interface FunctionLoss {
    float findAccuracy(NNArray outputs, NNArray idealOutputs);

    void findDerivative(NNArray outputs, NNArray idealOutputs, NNArray error);

    @NoArgsConstructor
    class Quadratic implements FunctionLoss {
        private float n = 2.0f;
        private float nDiv = 1;

        public Quadratic(double n) {
            this.n = (float) n;
            nDiv = (float) (2.0 / n);
        }

        @Override
        public float findAccuracy(NNArray outputs, NNArray idealOutputs) {
            return outputs.subPow2(idealOutputs);
        }

        @Override
        public void findDerivative(NNArray outputs, NNArray idealOutputs, NNArray error) {
            error.sub(outputs, idealOutputs);
            if (nDiv != 1) {
                error.mul(nDiv);
            }
        }
    }

    class BinaryCrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray outputs, NNArray idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.getSize(); i++) {
               accuracy -= idealOutputs.get(i) * log(outputs.get(i) + 0.00000001f) +
                       (1 - idealOutputs.get(i)) * log(1.0000001f - outputs.get(i));
            }
            return accuracy;
        }

        @Override
        public void findDerivative(NNArray outputs, NNArray idealOutputs, NNArray error) {
            error.binaryCrossEntropy(outputs, idealOutputs);
        }
    }

    class CrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray outputs, NNArray idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.getSize(); i++) {
               accuracy += -idealOutputs.get(i) * log(outputs.get(i) + 0.00000001f);
            }
            return accuracy;
        }

        @Override
        public void findDerivative(NNArray outputs, NNArray idealOutputs, NNArray error) {
            error.crossEntropy(outputs, idealOutputs);
        }
    }
}