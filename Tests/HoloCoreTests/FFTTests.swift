import XCTest
@testable import HoloCore

final class FFTTests: XCTestCase {
    func testMatchesDirectTransformForBroadbandSignal() {
        let input: [Double] = (0..<1_024).map { index in
            let phase = Double(index)
            return sin(phase * 0.373) + 0.5 * cos(phase * 1.117) + 0.25 * sin(phase * 2.719)
        }
        assertMatchesReference(input, size: 1_024)
    }

    func testMatchesDirectTransformForDecayingSinusoid() {
        let sampleRate = 48_000.0
        let input: [Double] = (0..<2_048).map { index in
            let time = Double(index) / sampleRate
            return 0.15 * exp(-time * 55) * sin(2 * .pi * 880 * time)
        }
        assertMatchesReference(input, size: 2_048)
    }

    func testMatchesDirectTransformWhenInputIsLongerThanSize() {
        let input = (0..<100).map { Double($0) * 0.01 }
        assertMatchesReference(input, size: 8)
    }

    func testMatchesDirectTransformWhenSizeIsNotAPowerOfTwo() {
        let input: [Double] = [1, 2, 3, 4, 5, 6]
        let spectrum = Radix2FFT.powerSpectrum(input, size: 6)
        XCTAssertEqual(spectrum.count, 5)
        assertMatchesReference(input, size: 6)
    }

    func testImpulseProducesFlatSpectrum() {
        var input = [Double](repeating: 0, count: 1_024)
        input[512] = 1
        let window = 0.5 - 0.5 * cos(2 * Double.pi * 512 / 1_023)
        let expected = window * window / 1_024

        let spectrum = Radix2FFT.powerSpectrum(input, size: 1_024)
        for (bin, value) in spectrum.enumerated() {
            XCTAssertEqual(value, expected, accuracy: expected * 1e-6, "bin \(bin)")
        }
    }

    func testConstantSignalConcentratesEnergyAtDC() {
        let input = [Double](repeating: 1, count: 1_024)
        let spectrum = Radix2FFT.powerSpectrum(input, size: 1_024)
        XCTAssertEqual(spectrum.firstIndex(of: spectrum.max() ?? 0), 0)
    }

    func testSinusoidAtExactBinPeaksAtThatBin() {
        let input: [Double] = (0..<1_024).map { index in
            sin(2 * Double.pi * 64 * Double(index) / 1_024)
        }
        let spectrum = Radix2FFT.powerSpectrum(input, size: 1_024)
        XCTAssertEqual(spectrum.firstIndex(of: spectrum.max() ?? 0), 64)
    }

    func testEmptyInputProducesTwoZeroBins() {
        XCTAssertEqual(Radix2FFT.powerSpectrum([]), [0, 0])
    }

    func testSingleSampleProducesTwoZeroBins() {
        // A one-sample input has no window support (`copied == 1`), so the
        // transform input is all zeros. Preserved from the original transform.
        XCTAssertEqual(Radix2FFT.powerSpectrum([5]), [0, 0])
    }

    func testOutputLengthIsAlwaysHalfSizePlusOne() {
        let input = [Double](repeating: 1, count: 32)
        for size in [8, 64, 4_096] {
            XCTAssertEqual(Radix2FFT.powerSpectrum(input, size: size).count, size / 2 + 1)
        }
    }

    /// Direct O(n^2) DFT applying the same window and scaling conventions as
    /// `Radix2FFT.powerSpectrum`, kept independent of the implementation under
    /// test so the FFT math is validated regardless of windowing choices.
    private func referencePowerSpectrum(_ input: [Double], size requestedSize: Int? = nil) -> [Double] {
        let desired = requestedSize ?? input.count
        let size = Radix2FFT.nextPowerOfTwo(max(desired, 2))
        let copied = min(input.count, size)

        var windowed = [Double](repeating: 0, count: size)
        if copied > 1 {
            for index in 0..<copied {
                let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(copied - 1))
                windowed[index] = input[index] * window
            }
        }

        return (0...(size / 2)).map { bin in
            var real = 0.0
            var imaginary = 0.0
            for index in 0..<size {
                let angle = -2 * Double.pi * Double(bin) * Double(index) / Double(size)
                real += windowed[index] * cos(angle)
                imaginary += windowed[index] * sin(angle)
            }
            return (real * real + imaginary * imaginary) / Double(size)
        }
    }

    private func assertMatchesReference(
        _ input: [Double],
        size: Int? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let actual = Radix2FFT.powerSpectrum(input, size: size)
        let expected = referencePowerSpectrum(input, size: size)
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for index in actual.indices {
            let tolerance = max(1e-9, abs(expected[index]) * 1e-6)
            XCTAssertEqual(
                actual[index],
                expected[index],
                accuracy: tolerance,
                "bin \(index)",
                file: file,
                line: line
            )
        }
    }
}
