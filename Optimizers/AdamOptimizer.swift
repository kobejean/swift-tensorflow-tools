//
//  Adam.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

@usableFromInline
class AdamOptimizer<Parameters: ParameterGroup>: Optimizer where Parameters.Parameter == Tensor<Float> {
    typealias Scalar = Float
    let α: Scalar
    let βm: Scalar
    let βv: Scalar
    let ϵ: Scalar
    var m: Parameters
    var v: Parameters
    
    init(_ θ: Parameters, _ α: Scalar = 0.001, _ βm: Scalar = 0.9, _ βv: Scalar = 0.999, _ ϵ: Scalar = 1e-08) {
        self.α = α
        self.βm = βm
        self.βv = βv
        self.ϵ = ϵ
        var zero = θ
        zero.update(withGradients: θ) { (m_k, θ_k) in m_k = 0 * θ_k }
        self.m = zero
        self.v = zero
    }
    
    @usableFromInline
    func optimize(_ θ: inout Parameters, _ dθ: Parameters) {
        m.update(withGradients: dθ) { (mk, dθk) in mk = βm * mk + (1 - βm) * dθk }
        v.update(withGradients: dθ) { (vk, dθk) in vk = βv * vk + (1 - βv) * dθk * dθk }
        var θ_delta = m
        θ_delta.update(withGradients: v) { (θk_delta, vk) in
            let mk = θk_delta
            let m_hat = mk / (1 - βm)
            let v_hat = vk / (1 - βv)
            θk_delta = α * m_hat / (√v_hat + ϵ)
        }
        θ.update(withGradients: θ_delta, -=)
    }
}



