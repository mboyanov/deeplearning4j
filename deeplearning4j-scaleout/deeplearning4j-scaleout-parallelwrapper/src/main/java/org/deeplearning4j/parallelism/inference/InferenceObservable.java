package org.deeplearning4j.parallelism.inference;

import org.deeplearning4j.parallelism.inference.observers.Batch;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Observer;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    Batch getInput();

    void setInput(INDArray... input);
    
    void setInput(INDArray[] input, INDArray[] masks);

    void setOutput(INDArray... output);

    void addObserver(Observer observer);

    INDArray[] getOutput();
}
