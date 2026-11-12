import AVFoundation
import CryptoKit
import Foundation

/// Per-take metadata, schema v1 (docs/CAPTURE_APP.md "Metadata JSON"). Every key is always
/// written; optional values are written as JSON null, never omitted.
struct TakeMetadata: Encodable {
    let schemaVersion: Int
    let take: String
    let startedUTC: String
    let durationS: Double
    let txFile: String?        // null for "none (no playback)" (D-088)
    let txChannel: TxChannel
    let txSha256: String?      // null for "none (no playback)" (D-088)
    let route: Route
    let micSource: MicSource
    let sampleRate: Int
    let ioBufferS: Double
    let inputLatencyS: Double
    let outputLatencyS: Double
    let dataSourceName: String?
    let polarPattern: String?
    let iosVersion: String
    let deviceModel: String
    let label: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case take
        case startedUTC = "started_utc"
        case durationS = "duration_s"
        case txFile = "tx_file"
        case txChannel = "tx_channel"
        case txSha256 = "tx_sha256"
        case route
        case micSource = "mic_source"
        case sampleRate = "sample_rate"
        case ioBufferS = "io_buffer_s"
        case inputLatencyS = "input_latency_s"
        case outputLatencyS = "output_latency_s"
        case dataSourceName = "data_source_name"
        case polarPattern = "polar_pattern"
        case iosVersion = "ios_version"
        case deviceModel = "device_model"
        case label
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schemaVersion, forKey: .schemaVersion)
        try c.encode(take, forKey: .take)
        try c.encode(startedUTC, forKey: .startedUTC)
        try c.encode(durationS, forKey: .durationS)
        // Explicit null for a no-playback take (encodeIfPresent would drop the key).
        if let v = txFile { try c.encode(v, forKey: .txFile) } else { try c.encodeNil(forKey: .txFile) }
        try c.encode(txChannel, forKey: .txChannel)
        if let v = txSha256 { try c.encode(v, forKey: .txSha256) } else { try c.encodeNil(forKey: .txSha256) }
        try c.encode(route, forKey: .route)
        try c.encode(micSource, forKey: .micSource)
        try c.encode(sampleRate, forKey: .sampleRate)
        try c.encode(ioBufferS, forKey: .ioBufferS)
        try c.encode(inputLatencyS, forKey: .inputLatencyS)
        try c.encode(outputLatencyS, forKey: .outputLatencyS)
        // Explicit null when unavailable (encodeIfPresent would drop the key).
        if let v = dataSourceName { try c.encode(v, forKey: .dataSourceName) } else { try c.encodeNil(forKey: .dataSourceName) }
        if let v = polarPattern { try c.encode(v, forKey: .polarPattern) } else { try c.encodeNil(forKey: .polarPattern) }
        try c.encode(iosVersion, forKey: .iosVersion)
        try c.encode(deviceModel, forKey: .deviceModel)
        try c.encode(label, forKey: .label)
    }

    static let schemaVersion = 1
}

/// Files and helpers for takes in the app's Documents directory.
enum Recording {
    static let sampleRate: Double = 48_000

    /// Labels must be non-empty and use only these characters (lead recommendation, Q1b).
    static let labelCharacters = CharacterSet(charactersIn:
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")

    static func isValidLabel(_ label: String) -> Bool {
        !label.isEmpty && label.unicodeScalars.allSatisfy { labelCharacters.contains($0) }
    }

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func wavURL(_ take: String) -> URL { documents.appendingPathComponent(take + ".wav") }
    static func jsonURL(_ take: String) -> URL { documents.appendingPathComponent(take + ".json") }

    /// `<label>`, or `<label>_2`, `<label>_3`, ... if a take with that stem already exists.
    static func uniqueTakeName(for label: String) -> String {
        let fm = FileManager.default
        func taken(_ stem: String) -> Bool {
            fm.fileExists(atPath: wavURL(stem).path) || fm.fileExists(atPath: jsonURL(stem).path)
        }
        if !taken(label) { return label }
        var n = 2
        while taken("\(label)_\(n)") { n += 1 }
        return "\(label)_\(n)"
    }

    /// Saved take stems (one per `<take>.wav`), newest first.
    static func savedTakes() -> [String] {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: documents,
                                                     includingPropertiesForKeys: [.contentModificationDateKey]) else {
            return []
        }
        let wavs = urls.filter { $0.pathExtension.lowercased() == "wav" }
        let dated = wavs.map { url -> (String, Date) in
            let d = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return (url.deletingPathExtension().lastPathComponent, d)
        }
        return dated.sorted { $0.1 > $1.1 }.map { $0.0 }
    }

    /// Deletes `<take>.wav` and `<take>.json`; logs each result.
    static func deleteTake(_ take: String) {
        for url in [wavURL(take), jsonURL(take)] {
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            do {
                try FileManager.default.removeItem(at: url)
                dlog("deleted \(url.lastPathComponent)")
            } catch {
                dlog("delete \(url.lastPathComponent) FAILED: \(error.localizedDescription)")
            }
        }
    }

    /// tx WAVs bundled from Resources/tx/. XcodeGen adds Resources as a group, so Xcode copies
    /// the files flat into the bundle root (docs/CAPTURE_APP.md).
    static func bundledTxFiles() -> [String] {
        let urls = Bundle.main.urls(forResourcesWithExtension: "wav", subdirectory: nil) ?? []
        return urls.map { $0.lastPathComponent }.sorted()
    }

    static func bundledTxURL(_ name: String) -> URL? {
        let stem = (name as NSString).deletingPathExtension
        return Bundle.main.url(forResource: stem, withExtension: "wav")
    }

    /// SHA-256 of the tx WAV file bytes, lowercase hex (lead recommendation, Q2).
    static func sha256Hex(of url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// `utsname.machine`, e.g. "iPhone18,1". Never hardcoded.
    static func deviceModel() -> String {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    static func utcStamp(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]   // "2026-10-20T01:02:03Z"
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: date)
    }

    static func writeMetadata(_ meta: TakeMetadata, take: String) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        let data = try enc.encode(meta)
        try data.write(to: jsonURL(take), options: .atomic)
    }

    /// Output file: float32, 48 kHz, mono WAV.
    static func makeOutputFile(_ url: URL) throws -> AVAudioFile {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        return try AVAudioFile(forWriting: url, settings: settings,
                               commonFormat: .pcmFormatFloat32, interleaved: false)
    }
}

/// Writes input-tap buffers (channel 0 only) into the take's WAV until `targetFrames` are written.
/// Called from the tap's thread; `finish` from the main thread. All state is behind one lock.
final class TakeWriter: @unchecked Sendable {
    /// Stop rule (Q3, pending the user's answer). true: the last chunk is cut so the file holds
    /// exactly `targetFrames`. false: the chunk that crosses `targetFrames` is written whole.
    static let truncateToTarget = true

    private let lock = NSLock()
    private var file: AVAudioFile?
    private let monoFormat: AVAudioFormat
    private let targetFrames: AVAudioFramePosition
    private var framesWritten: AVAudioFramePosition = 0
    private var done = false
    private var error: Error?

    init(file: AVAudioFile, targetFrames: AVAudioFramePosition) {
        self.file = file
        self.monoFormat = file.processingFormat
        self.targetFrames = targetFrames
    }

    /// Appends channel 0 of `buffer`. Returns true exactly once: when the target is reached or a
    /// write fails. After that every call is a no-op.
    func append(_ buffer: AVAudioPCMBuffer) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !done, let file = file else { return false }
        let available = AVAudioFramePosition(buffer.frameLength)
        guard available > 0, let src = buffer.floatChannelData?[0] else { return false }
        var n = available
        if TakeWriter.truncateToTarget {
            n = min(available, targetFrames - framesWritten)
        }
        guard n > 0, let mono = AVAudioPCMBuffer(pcmFormat: monoFormat,
                                                 frameCapacity: AVAudioFrameCount(n)),
              let dst = mono.floatChannelData?[0] else { return false }
        // Interleaved input: channel 0 is every channelCount-th sample.
        let stride = buffer.format.isInterleaved ? Int(buffer.format.channelCount) : 1
        for i in 0..<Int(n) {
            dst[i] = src[i * stride]
        }
        mono.frameLength = AVAudioFrameCount(n)
        do {
            try file.write(from: mono)
        } catch {
            self.error = error
            done = true
            return true
        }
        framesWritten += n
        if framesWritten >= targetFrames {
            done = true
            return true
        }
        return false
    }

    /// Stops writing and releases the AVAudioFile so its WAV header is finalized (iOS 17 has no
    /// `close()`). Returns frames written and the write error, if any.
    func finish() -> (frames: AVAudioFramePosition, error: Error?) {
        lock.lock()
        defer { lock.unlock() }
        done = true
        file = nil
        return (framesWritten, error)
    }
}
