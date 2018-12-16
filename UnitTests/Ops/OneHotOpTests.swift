//
//  OneHotOpTests.swift
//  SwiftForTensorFlowToolsUnitTests
//
//  Created by Jean Flaherty on 12/15/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

@testable import SwiftForTensorFlowTools
import TensorFlow
import XCTest

class OneHotOpTests: XCTestCase {

    func testOneHot() {
        let x = Tensor<Int32>([0,1,2,3,4])
        let expected = Tensor<Float>([[1,0,0,0,0],
                                      [0,1,0,0,0],
                                      [0,0,1,0,0],
                                      [0,0,0,1,0],
                                      [0,0,0,0,1]])
        let result: Tensor<Float> = e_i(x,5)
        XCTAssertEqual(result, expected)
    }

}
