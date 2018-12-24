//
//  Types.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

public typealias TensorFlowInteger = BinaryInteger & TensorFlowScalar

public protocol NumericTensor {
    associatedtype Scalar
    
    static func + (lhs: Self, rhs: Self) -> Self
    static func + (lhs: Self, rhs: Scalar) -> Self
    static func + (lhs: Scalar, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Scalar) -> Self
    static func - (lhs: Scalar, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Scalar) -> Self
    static func * (lhs: Scalar, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Scalar) -> Self
    static func / (lhs: Scalar, rhs: Self) -> Self
    
    static func += (lhs: inout Self, rhs: Self)
    static func += (lhs: inout Self, rhs: Scalar)
    static func -= (lhs: inout Self, rhs: Self)
    static func -= (lhs: inout Self, rhs: Scalar)
    static func *= (lhs: inout Self, rhs: Self)
    static func *= (lhs: inout Self, rhs: Scalar)
    static func /= (lhs: inout Self, rhs: Self)
    static func /= (lhs: inout Self, rhs: Scalar)
}
public protocol BinaryFloatingPointTensor: NumericTensor {
    static prefix func √ (rhs: Self) -> Self
}

extension Tensor: NumericTensor where Tensor.Scalar: Numeric { }
extension Tensor: BinaryFloatingPointTensor where Tensor.Scalar: BinaryFloatingPoint { }
