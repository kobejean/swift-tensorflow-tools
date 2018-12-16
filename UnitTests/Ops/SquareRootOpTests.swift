//
//  SquareRootOpTests.swift
//  SwiftForTensorFlowToolsUnitTests
//
//  Created by Jean Flaherty on 12/15/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

@testable import SwiftForTensorFlowTools
import TensorFlow
import XCTest

class SquareRootOpTests: XCTestCase {

    func testSquareRoot() {
        let x = Tensor<Double>([1,4,9,16,25,36,49,64,81,100])
        let expected = Tensor<Double>([1,2,3,4,5,6,7,8,9,10])
        let result = √x
        XCTAssertEqual(result, expected)
    }
    
    func testSquareRootPerformance256x256() {
        var x = Tensor<Float>(randomUniform: [256, 256])
        self.measure { √x }
    }
    
    func testSquareRootPerformance512x512() {
        let x = Tensor<Float>(randomUniform: [512, 512])
        self.measure { √x }
    }

}
