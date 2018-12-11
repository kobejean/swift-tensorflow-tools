//
//  Ops.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/2/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

@inlinable @inline(__always)
public func oneHot<T: TensorFlowScalar, Ti: TensorFlowInteger>(indices: Tensor<Ti>, depth: Int32, onValue: T, offValue: T, axis: Int64 = -1) -> Tensor<T> {
    let depthTensor = Tensor<Int32>(depth)
    let onValueTensor = Tensor<T>(onValue)
    let offValueTensor = Tensor<T>(offValue)
    return Raw.oneHot(indices: indices, depth: depthTensor, onValue: onValueTensor, offValue: offValueTensor)
}
