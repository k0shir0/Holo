import Foundation

/// A low-allocation streaming onset detector. It adapts to the local noise floor,
/// retains a short pre-roll, and emits a fixed-length analysis window.
public final class StreamingTapDetector {
    public let sampleRate: Double
    public let channelCount: Int
    public let analysisWindowSamples: Int
    public let preRollSamples: Int
    public let warmUpSamples: Int

    public private(set) var noiseFloorRMS: Double
    public private(set) var totalSamples: Int64 = 0

    private let initialNoiseFloorRMS: Double
    private var preRoll: [[Float]]
    private var capture: [[Float]]?
    private var captureOnsetOffset = 0
    private var captureStreamIndex: Int64 = 0
    private var captureNoiseFloor: Double = 0
    private var refractorySamplesRemaining = 0
    private var adaptNoiseDuringRefractory = false
    private var warmUpSamplesRemaining: Int
    private var onsetFilter: OnePoleCascadeFilter<Float>

    public init(
        sampleRate: Double,
        channelCount: Int,
        analysisDuration: Double = 0.090,
        preRollDuration: Double = 0.012,
        warmUpDuration: Double = 0.75,
        initialNoiseFloorRMS: Double = 0.0005
    ) {
        self.sampleRate = sampleRate
        self.channelCount = max(channelCount, 1)
        self.analysisWindowSamples = max(Int(sampleRate * analysisDuration), 1_024)
        self.preRollSamples = max(Int(sampleRate * preRollDuration), 128)
        self.warmUpSamples = max(Int(sampleRate * warmUpDuration), 0)
        self.initialNoiseFloorRMS = max(initialNoiseFloorRMS, 0.000_01)
        self.noiseFloorRMS = max(initialNoiseFloorRMS, 0.000_01)
        self.warmUpSamplesRemaining = max(Int(sampleRate * warmUpDuration), 0)
        self.preRoll = Array(repeating: [], count: max(channelCount, 1))
        self.onsetFilter = OnePoleCascadeFilter(sampleRate: sampleRate)
    }

    public func reset() {
        totalSamples = 0
        noiseFloorRMS = initialNoiseFloorRMS
        preRoll = Array(repeating: [], count: channelCount)
        capture = nil
        refractorySamplesRemaining = 0
        adaptNoiseDuringRefractory = false
        warmUpSamplesRemaining = warmUpSamples
        onsetFilter.reset()
    }

    public func process(channels incoming: [[Float]]) -> [DetectedTap] {
        guard !incoming.isEmpty else { return [] }
        let frameCount = incoming.map(\.count).min() ?? 0
        guard frameCount > 0 else { return [] }

        let channels = normalizedChannels(incoming, frameCount: frameCount)
        let mono = mixDown(channels)
        let onsetSignal = lowPassForOnset(mono)
        defer { totalSamples += Int64(frameCount) }

        let rms = rootMeanSquare(onsetSignal)
        let peak = onsetSignal.map { abs(Double($0)) }.max() ?? 0

        // The initial fixed floor is only a safe bootstrap value. A MacBook mic
        // in a real room can sit well above it; trying to detect before learning
        // that floor creates a loop where every buffer looks like an impulse and
        // the floor never gets a chance to rise.
        if warmUpSamplesRemaining > 0 {
            adaptNoiseFloor(to: rms, isWarmUp: true)
            warmUpSamplesRemaining = max(0, warmUpSamplesRemaining - frameCount)
            appendToPreRoll(channels)
            return []
        }

        if refractorySamplesRemaining > 0 {
            if adaptNoiseDuringRefractory {
                adaptNoiseFloor(to: rms, isWarmUp: false)
            }
            refractorySamplesRemaining = max(0, refractorySamplesRemaining - frameCount)
            if refractorySamplesRemaining == 0 {
                adaptNoiseDuringRefractory = false
            }
            appendToPreRoll(channels)
            return []
        }

        if capture != nil {
            appendToCapture(channels)
            if let event = completeCaptureIfReady() {
                return [event]
            }
            return []
        }

        // Detect on a low-pass signal so the optional 15.5–21 kHz probe
        // cannot arm its own capture. The emitted event still contains the
        // untouched full-band channels used by the feature extractor.
        let rmsThreshold = max(noiseFloorRMS * 1.18, 0.0008)
        let peakThreshold = max(noiseFloorRMS * 4.0, 0.007)
        let crest = peak / max(rms, 0.000_001)
        let strongSampleThreshold = max(peakThreshold, peak * 0.55)
        let strongSampleFraction = Double(onsetSignal.filter {
            abs(Double($0)) >= strongSampleThreshold
        }.count) / Double(max(frameCount, 1))
        let isImpulse = rms > rmsThreshold
            && peak > peakThreshold
            && crest > 2.0
            && strongSampleFraction < 0.20

        if isImpulse {
            let crossing = onsetSignal.firstIndex { abs(Double($0)) >= peakThreshold } ?? 0
            capture = preRoll
            captureOnsetOffset = (preRoll.first?.count ?? 0) + crossing
            captureStreamIndex = totalSamples + Int64(crossing)
            captureNoiseFloor = noiseFloorRMS
            appendToCapture(channels)
            if let event = completeCaptureIfReady() {
                return [event]
            }
        } else {
            adaptNoiseFloor(to: rms, isWarmUp: false)
            appendToPreRoll(channels)
        }

        return []
    }

    private func normalizedChannels(_ incoming: [[Float]], frameCount: Int) -> [[Float]] {
        var result = Array(repeating: Array(repeating: Float.zero, count: frameCount), count: channelCount)
        for channel in 0..<channelCount {
            let source = incoming[min(channel, incoming.count - 1)]
            result[channel] = Array(source.prefix(frameCount))
        }
        return result
    }

    private func mixDown(_ channels: [[Float]]) -> [Float] {
        guard channels.count > 1 else { return channels[0] }
        var mono = Array(repeating: Float.zero, count: channels[0].count)
        let scale = 1 / Float(channels.count)
        for channel in channels {
            for index in mono.indices { mono[index] += channel[index] * scale }
        }
        return mono
    }

    private func rootMeanSquare(_ values: [Float]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sum = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        return sqrt(sum / Double(values.count))
    }

    private func adaptNoiseFloor(to measuredRMS: Double, isWarmUp: Bool) {
        guard measuredRMS.isFinite else { return }
        let measured = max(measuredRMS, 0.000_01)
        if isWarmUp {
            let alpha = 0.14
            noiseFloorRMS = (1 - alpha) * noiseFloorRMS + alpha * measured
            return
        }

        // Track sustained room changes quickly enough to avoid a false-trigger
        // cascade, but cap one-step growth so a real tap does not become the new
        // baseline. Downward movement is deliberately slower and steadier.
        let capped = min(measured, max(noiseFloorRMS * 3.5, 0.020))
        let alpha = capped > noiseFloorRMS ? 0.06 : 0.025
        noiseFloorRMS = (1 - alpha) * noiseFloorRMS + alpha * capped
    }

    private func lowPassForOnset(_ values: [Float]) -> [Float] {
        // Persistent state: the filter carries continuity across streaming buffers.
        onsetFilter.process(values)
    }

    private func appendToPreRoll(_ channels: [[Float]]) {
        for index in 0..<channelCount {
            preRoll[index].append(contentsOf: channels[index])
            if preRoll[index].count > preRollSamples {
                preRoll[index].removeFirst(preRoll[index].count - preRollSamples)
            }
        }
    }

    private func appendToCapture(_ channels: [[Float]]) {
        guard var current = capture else { return }
        for index in 0..<channelCount {
            current[index].append(contentsOf: channels[index])
        }
        capture = current
    }

    private func completeCaptureIfReady() -> DetectedTap? {
        guard let current = capture, (current.first?.count ?? 0) >= analysisWindowSamples else {
            return nil
        }
        let trimmed = current.map { Array($0.prefix(analysisWindowSamples)) }
        let event = DetectedTap(
            channels: trimmed,
            onsetOffset: min(captureOnsetOffset, analysisWindowSamples - 1),
            streamSampleIndex: captureStreamIndex,
            noiseFloorRMS: captureNoiseFloor
        )
        let accepted = ImpactEventGate.accepts(event, sampleRate: sampleRate)
        capture = nil
        preRoll = Array(repeating: [], count: channelCount)
        refractorySamplesRemaining = Int(sampleRate * 0.14)
        // A rejected sustained event is likely speech or a changed background.
        // Let the floor follow it during the refractory period so conversation
        // cannot repeatedly re-arm the detector every 140 ms.
        adaptNoiseDuringRefractory = !accepted
        return accepted ? event : nil
    }
}
