import Foundation

/// Rejects obviously sustained acoustic events after the streaming onset detector
/// has captured a complete candidate. This is deliberately a conservative gate:
/// ambiguous short sounds continue to the profile classifier, while speech-like
/// events with energy that remains high across most of the window are discarded.
enum ImpactEventGate {
    struct Metrics: Equatable, Sendable {
        var onsetContrast: Double
        var effectiveDurationSeconds: Double
        var earlyEnergyFraction: Double
        var lateToImpactRMS: Double
    }

    static func accepts(_ event: DetectedTap, sampleRate: Double) -> Bool {
        guard let metrics = metrics(for: event, sampleRate: sampleRate) else { return false }

        // A contact impact should emerge clearly from the immediately preceding
        // audio. This removes most consonant peaks that occur in the middle of
        // continuous speech without requiring a general-purpose speech model.
        guard metrics.onsetContrast >= 1.8 else { return false }

        // Effective duration at 40% of the peak envelope is a common
        // percussive-versus-sustained cue. Requiring all three sustained cues
        // keeps the gate generous toward desks with a long resonant tail.
        let isClearlySustained = metrics.effectiveDurationSeconds >= 0.040
            && metrics.lateToImpactRMS >= 0.36
            && metrics.earlyEnergyFraction < 0.60
        return !isClearlySustained
    }

    static func metrics(for event: DetectedTap, sampleRate: Double) -> Metrics? {
        guard sampleRate > 0,
              let first = event.channels.first,
              !first.isEmpty else { return nil }

        let frameCount = event.channels.map(\.count).min() ?? 0
        guard frameCount > 0 else { return nil }

        var mono = Array(repeating: 0.0, count: frameCount)
        let usableChannels = event.channels.filter { !$0.isEmpty }
        guard !usableChannels.isEmpty else { return nil }
        let channelScale = 1.0 / Double(usableChannels.count)
        for channel in usableChannels {
            for index in 0..<frameCount {
                mono[index] += Double(channel[index]) * channelScale
            }
        }

        // Match the detector's sub-6 kHz onset path so an optional ultrasonic
        // probe cannot make a real tap look artificially sustained.
        let filtered = lowPass(mono, sampleRate: sampleRate)
        let onset = min(max(event.onsetOffset, 0), filtered.count - 1)
        let frameSamples = max(Int(sampleRate * 0.005), 32)

        var frameRMS: [Double] = []
        var frameLengths: [Int] = []
        var offset = onset
        while offset < filtered.count {
            let end = min(offset + frameSamples, filtered.count)
            frameRMS.append(rms(filtered[offset..<end]))
            frameLengths.append(end - offset)
            offset = end
        }
        guard !frameRMS.isEmpty else { return nil }

        let impactSearchFrames = min(
            frameRMS.count,
            max(Int(ceil(0.025 * sampleRate / Double(frameSamples))), 1)
        )
        guard let impactRMS = frameRMS.prefix(impactSearchFrames).max(), impactRMS > 0 else {
            return nil
        }

        let preStart = max(0, onset - Int(sampleRate * 0.012))
        let preRMS = rms(filtered[preStart..<onset])
        let referenceFloor = max(preRMS, event.noiseFloorRMS, 0.000_001)
        let impactSearchEnd = min(onset + impactSearchFrames * frameSamples, filtered.count)
        let impactPeak = filtered[onset..<impactSearchEnd].map(abs).max() ?? 0
        // A fingertip impact may be only a few samples wide, especially over
        // elevated broadband room noise. Preserve those candidates even when
        // their 5 ms RMS is diluted by using a conservative peak alternative.
        let onsetContrast = max(
            impactRMS / referenceFloor,
            0.45 * impactPeak / referenceFloor
        )

        let effectiveThreshold = max(impactRMS * 0.40, event.noiseFloorRMS * 2.2)
        let effectiveSamples = zip(frameRMS, frameLengths).reduce(0) { partial, item in
            partial + (item.0 >= effectiveThreshold ? item.1 : 0)
        }
        let effectiveDuration = Double(effectiveSamples) / sampleRate

        let earlyEnd = min(onset + Int(sampleRate * 0.025), filtered.count)
        let totalEnergy = energy(filtered[onset..<filtered.count])
        let earlyEnergyFraction = energy(filtered[onset..<earlyEnd]) / max(totalEnergy, 1e-15)

        let lateStart = min(onset + Int(sampleRate * 0.040), filtered.count)
        let lateRMS = rms(filtered[lateStart..<filtered.count])

        return Metrics(
            onsetContrast: onsetContrast,
            effectiveDurationSeconds: effectiveDuration,
            earlyEnergyFraction: earlyEnergyFraction,
            lateToImpactRMS: lateRMS / max(impactRMS, 1e-12)
        )
    }

    private static func lowPass(_ values: [Double], sampleRate: Double) -> [Double] {
        // Fresh state per call: the gate filters one complete captured window.
        var filter = OnePoleCascadeFilter<Double>(sampleRate: sampleRate)
        return filter.process(values)
    }

    private static func rms(_ values: ArraySlice<Double>) -> Double {
        guard !values.isEmpty else { return 0 }
        return sqrt(energy(values) / Double(values.count))
    }

    private static func energy(_ values: ArraySlice<Double>) -> Double {
        values.reduce(0) { $0 + $1 * $1 }
    }
}
