//
//  SwiftForTensorFlowToolsUnitTests.swift
//  SwiftForTensorFlowToolsUnitTests
//
//  Created by Jean Flaherty on 12/15/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//
@testable import SwiftForTensorFlowTools
import TensorFlow
import XCTest

class PiProductUnitTests: XCTestCase {

    func testPiProduct() {
        let x = Tensor([1,2,3,4])
        let expected = Tensor([24])
        let result = ∏x
        XCTAssertEqual(result, expected)
    }
    
    func testPiProductPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { ∏x }
    }
    
    func testPiProductPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { ∏x }
    }

}
