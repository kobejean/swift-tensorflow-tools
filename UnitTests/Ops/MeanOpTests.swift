//
//  MeanOpTests.swift
//  SwiftForTensorFlowToolsUnitTests
//
//  Created by Jean Flaherty on 12/15/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

@testable import SwiftForTensorFlowTools
import TensorFlow
import XCTest

class MeanOpTests: XCTestCase {
    
    func testMean() {
        let x = Tensor<Float>([1,2,3,4])
        let expected: Float = 2.5
        let result = μ(x)
        XCTAssertEqual(result, expected)
    }
    
    func testMeanAlongMajorAxis() {
        let x = Tensor<Float>([[1,2],
                               [3,4]])
        let expected = Tensor<Float>([[2,3]])
        let result = μ(x, alongAxes: 0)
        XCTAssertEqual(result, expected)
    }
    
    func testMeanAlongMinorAxis() {
        let x = Tensor<Float>([[1,2],
                               [3,4]])
        let expected = Tensor<Float>([[1.5],
                                      [3.5]])
        let result = μ(x, alongAxes: 1)
        XCTAssertEqual(result, expected)
    }
    
    func testMeanReducingMajorAxis() {
        let x = Tensor<Float>([[1,2],
                               [3,4]])
        let expected = Tensor<Float>([2,3])
        let result = μ(x, reducingAxes: 0)
        XCTAssertEqual(result, expected)
    }
    
    func testMeanReducingMinorAxis() {
        let x = Tensor<Float>([[1,2],
                               [3,4]])
        let expected = Tensor<Float>([1.5,3.5])
        let result = μ(x, reducingAxes: 1)
        XCTAssertEqual(result, expected)
    }
    
    func testMeanPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { μ(x) }
    }
    
    func testMeanPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { μ(x) }
    }
    
    func testMeanAlongMajorAxisPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { μ(x, alongAxes: 0) }
    }
    
    func testMeanAlongMajorAxisPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { μ(x, alongAxes: 0) }
    }
    
    func testMeanAlongMinorAxisPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { μ(x, alongAxes: 1) }
    }
    
    func testMeanAlongMinorAxisPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { μ(x, alongAxes: 1) }
    }
    
    func testMeanReducingMajorAxisPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { μ(x, reducingAxes: 0) }
    }
    
    func testMeanReducingMajorAxisPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { μ(x, reducingAxes: 0) }
    }
    
    func testMeanReducingMinorAxisPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { μ(x, reducingAxes: 1) }
    }
    
    func testMeanReducingMinorAxisPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { μ(x, reducingAxes: 1) }
    }

}
