import Accelerate
import Foundation

enum Radix2FFT {
    static func powerSpectrum(_ input: [Double], size requestedSize: Int? = nil) -> [Double] {
        let desired = requestedSize ?? input.count
        let size = nextPowerOfTwo(max(desired, 2))
        let copied = min(input.count, size)

        // Same windowing as the previous hand-rolled transform: a symmetric Hann
        // window over the first `copied` samples only, `copied - 1` denominator,
        // with the zero-padded tail left untouched.
        var windowed = [Double](repeating: 0, count: size)
        if copied > 1 {
            let denominator = Double(copied - 1)
            for index in 0..<copied {
                let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(index) / denominator)
                windowed[index] = input[index] * window
            }
        }

        // vDSP's real FFT requires log2n >= 3. Smaller sizes only arise from
        // degenerate inputs; a direct transform keeps them exact.
        guard size >= 8 else { return directPowerSpectrum(windowed, size: size) }

        let log2n = vDSP_Length(size.trailingZeroBitCount)
        let halfSize = size / 2
        let setup = FFTSetupCache.shared.setup(forLog2n: log2n)

        var realPart = [Double](repeating: 0, count: halfSize)
        var imaginaryPart = [Double](repeating: 0, count: halfSize)
        var power = [Double](repeating: 0, count: halfSize + 1)
        // vDSP's forward real FFT is unnormalized and scaled by 2 relative to the
        // textbook DFT. The convention here has always been |X|^2 / size, so the
        // packed output is divided by 2^2 * size.
        let divisor = 4.0 * Double(size)

        realPart.withUnsafeMutableBufferPointer { realBuffer in
            imaginaryPart.withUnsafeMutableBufferPointer { imaginaryBuffer in
                var split = DSPDoubleSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                windowed.withUnsafeBufferPointer { inputBuffer in
                    inputBuffer.baseAddress!.withMemoryRebound(
                        to: DSPDoubleComplex.self,
                        capacity: halfSize
                    ) { complexInput in
                        vDSP_ctozD(complexInput, 2, &split, 1, vDSP_Length(halfSize))
                    }
                }
                vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                // Packed layout: realp[0] holds DC and imagp[0] holds Nyquist,
                // both purely real.
                power[0] = (realBuffer[0] * realBuffer[0]) / divisor
                for bin in 1..<halfSize {
                    power[bin] = (realBuffer[bin] * realBuffer[bin]
                        + imaginaryBuffer[bin] * imaginaryBuffer[bin]) / divisor
                }
                power[halfSize] = (imaginaryBuffer[0] * imaginaryBuffer[0]) / divisor
            }
        }
        return power
    }

    static func nextPowerOfTwo(_ value: Int) -> Int {
        var result = 1
        while result < value { result <<= 1 }
        return result
    }

    private static func directPowerSpectrum(_ windowed: [Double], size: Int) -> [Double] {
        (0...(size / 2)).map { bin in
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
}

/// Twiddle tables are expensive to build relative to one 2k/4k transform, and
/// the same sizes recur on every detected tap, so completed setups are kept for
/// the lifetime of the process (bounded: one per power of two actually used).
private final class FFTSetupCache: @unchecked Sendable {
    static let shared = FFTSetupCache()
    private let lock = NSLock()
    private var setups: [vDSP_Length: FFTSetupD] = [:]

    func setup(forLog2n log2n: vDSP_Length) -> FFTSetupD {
        lock.lock()
        defer { lock.unlock() }
        if let existing = setups[log2n] { return existing }
        guard let created = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            preconditionFailure("vDSP_create_fftsetupD failed for log2n=\(log2n)")
        }
        setups[log2n] = created
        return created
    }
}
