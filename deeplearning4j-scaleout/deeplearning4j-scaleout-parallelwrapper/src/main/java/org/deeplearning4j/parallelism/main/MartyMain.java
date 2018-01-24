package org.deeplearning4j.parallelism.main;

import java.io.IOException;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MartyMain {

    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/tmp/calculator.hdf5", false);
        model.getConfiguration().addPreProcessors(InputType.recurrent(1,3));
        ComputationGraph g2 = new ComputationGraph(model.getConfiguration());
        g2.init();


        for (int i =0; i< model.getLayers().length; i++) {
            Layer l = model.getLayers()[i];
            g2.getLayer(i).setParamTable(l.paramTable());
        }
        ParallelInference inf = new ParallelInference.Builder(g2).inferenceMode(InferenceMode.BATCHED).build();
        
        INDArray[] out = inf.output(new INDArray[] { Nd4j.ones(10,1, 3)}, new INDArray[]{null});
        System.out.println(out[0]);
        INDArray mask = Nd4j.zeros(10,3);
        for (int i =0; i<10 ; i++) {
            for (int j = 1; j<3; j++ ) {
                mask.putScalar(i, j, 1);
            }
        }
        out = inf.output(new INDArray[] { Nd4j.ones(10,1, 3)}, new INDArray[]{mask});
       
        System.out.println(out[0]);
    }
}
