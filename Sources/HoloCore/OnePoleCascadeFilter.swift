import Foundation

/// Four cascaded one-pole low-pass stages shared by the streaming onset detector
/// and the impact gate. Generic over the sample scalar so each call site keeps its
/// exact arithmetic: `Float` accumulates in single precision, `Double` in double.
struct OnePoleCascadeFilter<Scalar: BinaryFloatingPoint> {
    private let alpha: Scalar
    private var state: [Scalar]

    init(sampleRate: Double, stageCount: Int = 4) {
        let cutoff = min(6_000.0, sampleRate * 0.20)
        // Derived in Double exactly as both call sites did, then narrowed once.
        let alphaDouble = 1 - exp(-2 * Double.pi * cutoff / sampleRate)
        self.alpha = Scalar(alphaDouble)
        self.state = Array(repeating: Scalar.zero, count: stageCount)
    }

    mutating func reset() {
        for index in state.indices { state[index] = Scalar.zero }
    }

    mutating func process(_ values: [Scalar]) -> [Scalar] {
        guard !values.isEmpty else { return [] }
        var result = Array(repeating: Scalar.zero, count: values.count)
        for index in values.indices {
            var filtered = values[index]
            for stage in state.indices {
                state[stage] += alpha * (filtered - state[stage])
                filtered = state[stage]
            }
            result[index] = filtered
        }
        return result
    }
}
