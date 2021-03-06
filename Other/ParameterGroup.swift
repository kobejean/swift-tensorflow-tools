//
//  ParameterGroup.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

func combine<Parameters: ParameterGroup> (_ first: Parameters, _ second: Parameters, _ combiner: (Parameters.Parameter, Parameters.Parameter) -> Parameters.Parameter ) -> Parameters where Parameters.Parameter: NumericTensor {
    return first.updated(withGradients: second) { $0 = combiner($0, $1) }
}

extension ParameterGroup where Parameter: NumericTensor {
    
    func updated(withGradients: Self, _ updater: (inout Parameter, Parameter) -> Void) -> Self {
        var result = self
        result.update(withGradients: withGradients, updater)
        return result
    }
    
    static func + (lhs: Self, rhs: Self) -> Self { return lhs.updated(withGradients: rhs, +=) }
    static func - (lhs: Self, rhs: Self) -> Self { return lhs.updated(withGradients: rhs, -=) }
    static func * (lhs: Self, rhs: Self) -> Self { return lhs.updated(withGradients: rhs, *=) }
    static func / (lhs: Self, rhs: Self) -> Self { return lhs.updated(withGradients: rhs, /=) }
//    static func % (lhs: Self, rhs: Self) -> Self { return lhs.updated(withGradients: rhs, %=) }
    
    static func += (lhs: inout Self, rhs: Self) { lhs.update(withGradients: rhs, +=) }
    static func -= (lhs: inout Self, rhs: Self) { lhs.update(withGradients: rhs, -=) }
    static func *= (lhs: inout Self, rhs: Self) { lhs.update(withGradients: rhs, *=) }
    static func /= (lhs: inout Self, rhs: Self) { lhs.update(withGradients: rhs, /=) }
//    static func %= (lhs: inout Self, rhs: Self) { lhs.update(withGradients: rhs, %=) }
}
