package nnarray;

import lombok.Getter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NNVector extends NNArray {
    @Getter
    private final int length;

    public NNVector(int length) {
        super(length);
        this.length = length;
    }

    public NNVector(float[] data) {
        super(data.length);
        this.data = data;
        this.length = data.length;
    }

    public NNVector(NNVector vector) {
        this(vector.length);
    }

    public void save(FileWriter writer) throws IOException {
        writer.write(length + "\n");
        for (int i = 0; i < length; i++) {
            writer.write(data[i] + " ");
            if (i % 1000 == 0) {
                writer.flush();
            }
        }
        writer.write("\n");
        writer.flush();
    }

    public static NNVector read(Scanner scanner) {
        NNVector vector = new NNVector(Integer.parseInt(scanner.nextLine()));
        double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
        for (int j = 0; j < vector.length; j++) {
            vector.data[j] = (float) arr[j];
        }
        return vector;
    }
}
