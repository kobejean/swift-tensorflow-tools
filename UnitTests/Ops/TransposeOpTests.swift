//
//  TransposeOpTests.swift
//  SwiftForTensorFlowToolsUnitTests
//
//  Created by Jean Flaherty on 12/15/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

@testable import SwiftForTensorFlowTools
import TensorFlow
import XCTest

class TransposeOpTests: XCTestCase {

    func testTranspose() {
        let x = Tensor<Float>([[1,2], [3,4]])
        let expected = Tensor<Float>([[1,3], [2,4]])
        let result = x⊺
        XCTAssertEqual(result, expected)
    }
    
    func testTransposePerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { x⊺ }
    }
    
    func testTransposePerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { x⊺ }
    }

}
