package neural_network.initialization;

import lombok.NoArgsConstructor;
import nnarray.NNArray;
import nnarray.NNMatrix;
import nnarray.NNTensor4D;

import java.util.Random;

public interface Initializer {
    void initialize(NNArray weight);

    @NoArgsConstructor
    class RandomNormal implements Initializer {
        private final java.util.Random random = new java.util.Random();
        private float range = 1;

        public RandomNormal(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNArray weight) {
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) (random.nextGaussian() * range));
            }
        }
    }

    class RandomUniform implements Initializer {
        private float range = 1;

        public RandomUniform(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNArray weight) {
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) ((Math.random() - 0.5) * range));
            }
        }
    }

    class XavierUniform implements Initializer {
        @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (Math.sqrt(6.0 / (((NNMatrix) weight).getRow() + ((NNMatrix) weight).getColumn())));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (Math.sqrt(6.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * (((NNTensor4D) weight).getDepth() + ((NNTensor4D) weight).getLength()))));
            }
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class XavierNormal implements Initializer {
        private final java.util.Random random = new java.util.Random();

        @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (Math.sqrt(2.0 / (((NNMatrix) weight).getRow() + ((NNMatrix) weight).getColumn())));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (Math.sqrt(2.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * (((NNTensor4D) weight).getDepth() + ((NNTensor4D) weight).getLength()))));
            }
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) (random.nextGaussian() * value));
            }
        }
    }

    class HeUniform implements Initializer {
        @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (Math.sqrt(6.0 / ((NNMatrix) weight).getRow()));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (Math.sqrt(6.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * ((NNTensor4D) weight).getDepth())));
            }
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class HeNormal implements Initializer {
        private final java.util.Random random = new java.util.Random();

        @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (Math.sqrt(2.0 / ((NNMatrix) weight).getRow()));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (Math.sqrt(2.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * ((NNTensor4D) weight).getDepth())));
            }
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) (random.nextGaussian() * value));
            }
        }
    }

    class LeCunUniform implements Initializer {
                @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (Math.sqrt(3.0 / ((NNMatrix) weight).getColumn()));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (Math.sqrt(3.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * ((NNTensor4D) weight).getLength())));
            }
            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class LeCunNormal implements Initializer {
        private final java.util.Random random = new java.util.Random();

        @Override
        public void initialize(NNArray weight) {
            float value = 1;
            if (weight instanceof NNMatrix) {
                value = (float) (1.0 / Math.sqrt(((NNMatrix) weight).getColumn()));
            } else if (weight instanceof NNTensor4D) {
                value = (float) (1.0 / Math.sqrt((((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
                        * ((NNTensor4D) weight).getDepth())));
            }

            for (int i = 0; i < weight.getSize(); i++) {
                weight.fill(i, (float) (random.nextGaussian() * value));
            }
        }
    }
}
