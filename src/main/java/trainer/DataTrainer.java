package trainer;

import data.mnist.DataLoader;
import data.mnist.Image1dData;
import neural_network.network.DeepNeuralNetwork;
import nnarray.NNMatrix;

public class DataTrainer {
    private DataLoader loader;

    private int sizeTrainEpoch;
    private int sizeTestEpoch;

    public DataTrainer(int sizeTrainEpoch, int sizeTestEpoch, DataLoader loader) {
        this.loader = loader;
        this.sizeTrainEpoch = sizeTrainEpoch;
        this.sizeTestEpoch = sizeTestEpoch;
    }

    public void train(DeepNeuralNetwork network, int sizeBatch, int countEpoch) {
        for (int i = 0; i < countEpoch; i++) {
            int counter = 0;
            double accuracy = 0;
            for (int j = 0; j < (int) Math.ceil(sizeTrainEpoch * 1.0 / sizeBatch); j++) {
                NNMatrix[] data = loader.getNextTrainData(sizeBatch);
                accuracy += network.train(data[0], data[1]);
                counter += DataMetric.qualityTop1(data[1], network.getOutputs());
            }
            System.out.println("\t\t\t" + (i + 1) + " ЕПОХА ");
            System.out.println("Результат навчального датасету: ");
            System.out.println("Відсоток правильних відповідей: " + String.format("%.2f", counter * 1.0 / sizeTrainEpoch * 100) + " %");
            System.out.println("Точність на навчальном датасеті: " + String.format("%.5f", accuracy / sizeTrainEpoch));

            NNMatrix[] data = loader.getNextTestData(sizeTestEpoch);
            counter = DataMetric.qualityTop1(data[1], network.query(data[0]));
            accuracy = network.accuracy(data[1]);
            System.out.println("Результат тренувального датасету: ");
            System.out.println("Відсоток правильних відповідей: " + String.format("%.2f", counter * 1.0 / sizeTestEpoch * 100) + " %");
            System.out.println("Точність на тренувальном датасеті: " + String.format("%.5f", accuracy / sizeTestEpoch));
            System.out.println();
        }
    }
}
