package org.deeplearning4j.parallelism.inference.observers;

import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.Getter;

public class Batch {

    @Getter
    private final INDArray[] input;
    @Getter
    private final INDArray[] mask;
    
    public Batch(INDArray[] input, INDArray[] mask) {
        super();
        this.input = input;
        this.mask = mask;
    }
}
