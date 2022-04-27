package trainer;

import nnarray.NNMatrix;

public class DataMetric {
    public static int qualityTop1(NNMatrix ideal, NNMatrix output){
        int counter = 0;
        for (int i = 0; i < ideal.getRow(); i++) {
            float maxI = ideal.get(i, 0);
            float indexMaxI = 0;
            float max = output.get(i, 0);
            float indexMax = 0;
            for (int j = 1; j < ideal.getColumn(); j++) {
                if(max < output.get(i, j)) {
                    max = output.get(i, j);
                    indexMax = j;
                }
                if(maxI < ideal.get(i, j)) {
                    maxI = ideal.get(i, j);
                    indexMaxI = j;
                }
            }
            if(indexMax == indexMaxI) {
                counter++;
            }
        }
        return counter;
    }
}
