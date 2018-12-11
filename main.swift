//
//  main.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

struct ModelParameters : ParameterGroup {
    var w1 = Tensor<Float>(randomNormal: [4, 4])
    var w2 = Tensor<Float>(randomNormal: [30, 10])
    var b1 = Tensor<Float>(zeros: [1, 30])
    var b2 = Tensor<Float>(zeros: [1, 10])
}

let θ = ModelParameters()

print(∑θ.w1)
print(∑θ.w1⊺)
