import AVFoundation
import Foundation
import UIKit

// Types and `configure` follow docs/CAPTURE_APP.md "Audio configuration (must match exactly)".
enum Route: String, Codable { case earpiece, bottomSpeaker }
enum MicSource: String, Codable { case front, bottom }
enum TxChannel: String, Codable { case left, right, both }   // `both` exists only for Protocol A

extension Route: CaseIterable, Identifiable { var id: String { rawValue } }
extension MicSource: CaseIterable, Identifiable { var id: String { rawValue } }
extension TxChannel: CaseIterable, Identifiable { var id: String { rawValue } }

enum RigError: Error, CustomStringConvertible {
    case sampleRate(Double)
    case noDataSource(MicSource)
    case notBuiltIn(String)
    case voiceProcessingOn
    case txFile(String)
    case engine(String)
    case micPermission

    var description: String {
        switch self {
        case .sampleRate(let r): return "sample rate \(r) Hz, need 48000 (rejected, never resampled)"
        case .noDataSource(let m): return "no built-in mic data source for \(m.rawValue)"
        case .notBuiltIn(let r): return "route is not built-in: \(r)"
        case .voiceProcessingOn: return "inputNode.isVoiceProcessingEnabled is true"
        case .txFile(let s): return "tx file: \(s)"
        case .engine(let s): return "engine: \(s)"
        case .micPermission: return "microphone permission not granted (Settings > RingCapture > Microphone)"
        }
    }
}

/// 2-channel float32 buffer, same length as `mono`; the mono samples go into the selected
/// channel(s), zeros in the other (D-032).
func makeTxBuffer(mono: AVAudioPCMBuffer, channel: TxChannel) -> AVAudioPCMBuffer {
    let fmt = AVAudioFormat(standardFormatWithSampleRate: mono.format.sampleRate, channels: 2)!
    let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: mono.frameLength)!
    out.frameLength = mono.frameLength
    let n = Int(mono.frameLength)
    let src = mono.floatChannelData![0]
    let l = out.floatChannelData![0]
    let r = out.floatChannelData![1]
    for i in 0..<n {
        l[i] = (channel == .left || channel == .both) ? src[i] : 0
        r[i] = (channel == .right || channel == .both) ? src[i] : 0
    }
    return out
}

/// Session config, engine, record/play. One take at a time; a fresh AVAudioEngine per take.
@MainActor
final class AudioRig: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var status = "idle"
    @Published private(set) var takes: [String] = Recording.savedTakes()

    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var writer: TakeWriter?
    private var current: PendingTake?
    private var observers: [NSObjectProtocol] = []
    private var engineObserver: NSObjectProtocol?

    private struct PendingTake {
        let take: String
        let label: String
        let startedUTC: String
        let txFile: String?
        let txChannel: TxChannel
        let txSha256: String?
        let route: Route
        let mic: MicSource
        let ioBufferS: Double
        let inputLatencyS: Double
        let outputLatencyS: Double
        let dataSourceName: String?
        let polarPattern: String?
        let routeSignature: String
    }

    init() {
        let nc = NotificationCenter.default
        let s = AVAudioSession.sharedInstance()
        observers.append(nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: s,
                                        queue: .main) { [weak self] n in
            let reason = (n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt) ?? 0
            guard let self else { return }
            Task { @MainActor in self.routeChanged(reason: reason) }
        })
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: s,
                                        queue: .main) { [weak self] n in
            let type = (n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt) ?? 0
            guard let self else { return }
            Task { @MainActor in self.interrupted(type: type) }
        })
        observers.append(nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification,
                                        object: s, queue: .main) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.abortTake("media services were reset") }
        })
        dlog("app start: iOS \(UIDevice.current.systemVersion), model \(Recording.deviceModel())")
        // Without permission the input delivers silence; ask once, refuse takes until granted.
        let perm = AVAudioApplication.shared.recordPermission
        switch perm {
        case .granted:
            dlog("record permission: granted")
        case .denied:
            dlog("record permission: denied")
        case .undetermined:
            dlog("record permission: undetermined, requesting")
            AVAudioApplication.requestRecordPermission { granted in
                dlog("record permission request: granted \(granted)")
            }
        @unknown default:
            dlog("record permission: unknown value")
        }
    }

    // MARK: session configuration (verbatim from the spec, plus log lines)

    func configure(route: Route, mic: MicSource) throws {
        let s = AVAudioSession.sharedInstance()
        let opts: AVAudioSession.CategoryOptions = (route == .bottomSpeaker) ? [.defaultToSpeaker] : []
        try s.setCategory(.playAndRecord, mode: .measurement, options: opts)   // no AGC/NS/EC
        dlog("setCategory playAndRecord, mode measurement, options \(opts.rawValue) (route \(route.rawValue))")
        try s.setPreferredSampleRate(48_000)
        dlog("setPreferredSampleRate 48000")
        try s.setPreferredIOBufferDuration(0.005)
        dlog("setPreferredIOBufferDuration 0.005")
        try s.setActive(true)
        dlog("setActive true: sampleRate \(s.sampleRate), ioBufferDuration \(s.ioBufferDuration)")
        guard s.sampleRate == 48_000 else { throw RigError.sampleRate(s.sampleRate) }  // reject, never resample
        let builtIn = s.availableInputs?.first { $0.portType == .builtInMic }
        try s.setPreferredInput(builtIn)
        dlog("setPreferredInput \(builtIn?.portName ?? "nil")")
        let want: AVAudioSession.Orientation = (mic == .front) ? .front : .bottom
        if let ds = builtIn?.dataSources?.first(where: { $0.orientation == want }) {
            if ds.supportedPolarPatterns?.contains(.omnidirectional) == true {
                try ds.setPreferredPolarPattern(.omnidirectional)
                dlog("setPreferredPolarPattern omnidirectional on \(ds.dataSourceName)")
            } else {
                dlog("data source \(ds.dataSourceName) has no omnidirectional pattern")
            }
            try builtIn?.setPreferredDataSource(ds)
            dlog("setPreferredDataSource \(ds.dataSourceName) (orientation \(want.rawValue))")
        } else { throw RigError.noDataSource(mic) }
    }

    // MARK: record

    /// Configures the session, starts the input tap, then loops the tx buffer on the player.
    /// The take stops itself once `durationS` of input has been written.
    func record(label: String, txFile: String?, txChannel: TxChannel, route: Route, mic: MicSource,
                durationS: Int) {
        guard !isRecording else { return }
        guard Recording.isValidLabel(label) else { fail("label must match [A-Za-z0-9_.-]+"); return }
        do {
            try start(label: label, txFile: txFile, txChannel: txChannel, route: route, mic: mic,
                      durationS: durationS)
        } catch {
            teardown()
            fail("\(error)")
        }
    }

    private func start(label: String, txFile: String?, txChannel: TxChannel, route: Route,
                       mic: MicSource, durationS: Int) throws {
        dlog("--- take '\(label)': tx \(txFile ?? "none") ch \(txChannel.rawValue), route \(route.rawValue), mic \(mic.rawValue), \(durationS) s")
        guard AVAudioApplication.shared.recordPermission == .granted else { throw RigError.micPermission }
        try configure(route: route, mic: mic)
        let s = AVAudioSession.sharedInstance()
        logRoute(s.currentRoute)
        try checkBuiltIn(s.currentRoute)

        // Actual selected data source and polar pattern (null in the JSON when unavailable, Q4).
        let selected = s.currentRoute.inputs.first?.selectedDataSource
        let dsName = selected?.dataSourceName
        let polar = selected?.selectedPolarPattern.map { $0.rawValue.lowercased() }
        dlog("selected data source \(dsName ?? "nil"), polar \(polar ?? "nil")")

        // txFile nil = "none (no playback)" (D-088): no player, tx_file / tx_sha256 null.
        var tx: (stereo: AVAudioPCMBuffer, sha: String)?
        if let txFile {
            tx = try loadTx(txFile, channel: txChannel)
        } else {
            dlog("tx none (no playback)")
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        guard !input.isVoiceProcessingEnabled else { throw RigError.voiceProcessingOn }
        let inFormat = input.outputFormat(forBus: 0)
        dlog("input native format: \(inFormat)")
        guard inFormat.sampleRate == 48_000 else { throw RigError.sampleRate(inFormat.sampleRate) }
        let outRate = engine.outputNode.outputFormat(forBus: 0).sampleRate
        guard outRate == 48_000 else { throw RigError.sampleRate(outRate) }

        var player: AVAudioPlayerNode?
        if let tx {
            let p = AVAudioPlayerNode()
            engine.attach(p)
            engine.connect(p, to: engine.mainMixerNode, format: tx.stereo.format)
            player = p
        }

        let take = Recording.uniqueTakeName(for: label)
        let wav = Recording.wavURL(take)
        let file = try Recording.makeOutputFile(wav)
        let target = AVAudioFramePosition(durationS) * AVAudioFramePosition(Recording.sampleRate)
        let writer = TakeWriter(file: file, targetFrames: target)
        dlog("writing \(wav.lastPathComponent), target \(target) frames")

        // Latencies read at Record start, diagnostics only (D-019, Q5).
        current = PendingTake(take: take, label: label, startedUTC: Recording.utcStamp(Date()),
                              txFile: txFile, txChannel: txChannel, txSha256: tx?.sha, route: route,
                              mic: mic, ioBufferS: s.ioBufferDuration, inputLatencyS: s.inputLatency,
                              outputLatencyS: s.outputLatency, dataSourceName: dsName,
                              polarPattern: polar, routeSignature: AudioRig.signature(s.currentRoute))
        self.engine = engine
        self.player = player
        self.writer = writer
        isRecording = true
        status = "recording \(take)"

        // Tap first, then the player, so the first tx frame is captured (D-019).
        let done: @Sendable () -> Void = { [weak self] in
            guard let self else { return }
            Task { @MainActor in self.finishTake() }
        }
        // A hardware reconfiguration stops the engine; the tap then goes quiet, so abort (D-073).
        engineObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.abortTake("engine configuration changed (engine stopped)") }
        }
        input.installTap(onBus: 0, bufferSize: 4800, format: inFormat,
                         block: AudioRig.makeTap(writer: writer, done: done))
        engine.prepare()
        do {
            try engine.start()
        } catch {
            throw RigError.engine("start failed: \(error.localizedDescription)")
        }
        if let player, let tx {
            player.scheduleBuffer(tx.stereo, at: nil, options: .loops, completionHandler: nil)
            player.play()
            dlog("engine started, player looping")
        } else {
            dlog("engine started, no playback")
        }
    }

    /// Loads a bundled tx WAV; it must be mono 48 kHz (the engine would otherwise resample it).
    /// Returns the D-032 stereo buffer and the SHA-256 of the file bytes.
    private func loadTx(_ txFile: String, channel: TxChannel) throws -> (stereo: AVAudioPCMBuffer, sha: String) {
        guard let txURL = Recording.bundledTxURL(txFile) else { throw RigError.txFile("\(txFile) not in bundle") }
        let txSha = try Recording.sha256Hex(of: txURL)
        let txIn = try AVAudioFile(forReading: txURL)
        guard txIn.fileFormat.sampleRate == 48_000 else { throw RigError.sampleRate(txIn.fileFormat.sampleRate) }
        guard txIn.fileFormat.channelCount == 1 else {
            throw RigError.txFile("\(txFile) has \(txIn.fileFormat.channelCount) channels, need 1")
        }
        guard let mono = AVAudioPCMBuffer(pcmFormat: txIn.processingFormat,
                                          frameCapacity: AVAudioFrameCount(txIn.length)) else {
            throw RigError.txFile("cannot allocate \(txIn.length) frames")
        }
        try txIn.read(into: mono)
        let stereo = makeTxBuffer(mono: mono, channel: channel)
        dlog("tx \(txFile): \(mono.frameLength) frames, sha256 \(txSha)")
        return (stereo, txSha)
    }

    /// Built outside the main actor so the tap block carries no actor isolation (it runs on
    /// the audio tap thread).
    nonisolated private static func makeTap(writer: TakeWriter,
                                            done: @escaping @Sendable () -> Void) -> AVAudioNodeTapBlock {
        return { buffer, _ in
            if writer.append(buffer) { done() }
        }
    }

    private func finishTake() {
        guard isRecording, let pending = current, let writer = writer else { return }
        teardown()
        let result = writer.finish()
        self.writer = nil
        current = nil
        isRecording = false
        if let err = result.error {
            dlog("WAV write FAILED: \(err.localizedDescription)")
            Recording.deleteTake(pending.take)
            fail("WAV write failed; take deleted")
            return
        }
        let durationS = Double(result.frames) / Recording.sampleRate   // Q3: actual frames / 48000
        let meta = TakeMetadata(schemaVersion: TakeMetadata.schemaVersion, take: pending.take,
                                startedUTC: pending.startedUTC, durationS: durationS,
                                txFile: pending.txFile, txChannel: pending.txChannel,
                                txSha256: pending.txSha256, route: pending.route,
                                micSource: pending.mic, sampleRate: Int(Recording.sampleRate),
                                ioBufferS: pending.ioBufferS, inputLatencyS: pending.inputLatencyS,
                                outputLatencyS: pending.outputLatencyS,
                                dataSourceName: pending.dataSourceName,
                                polarPattern: pending.polarPattern,
                                iosVersion: UIDevice.current.systemVersion,
                                deviceModel: Recording.deviceModel(), label: pending.label)
        do {
            try Recording.writeMetadata(meta, take: pending.take)
            dlog("wrote \(pending.take).wav (\(result.frames) frames, \(durationS) s) + \(pending.take).json")
            status = "saved \(pending.take)"
        } catch {
            dlog("JSON write FAILED: \(error.localizedDescription)")
            Recording.deleteTake(pending.take)
            fail("JSON write failed; take deleted")
        }
        refreshTakes()
    }

    /// Aborts the running take: stops audio, deletes the partial files, logs why (Q6b).
    private func abortTake(_ why: String) {
        guard isRecording, let pending = current else { return }
        teardown()
        _ = writer?.finish()
        writer = nil
        current = nil
        isRecording = false
        dlog("take \(pending.take) ABORTED: \(why)")
        Recording.deleteTake(pending.take)
        status = "aborted: \(why)"
        refreshTakes()
    }

    private func teardown() {
        if let token = engineObserver {
            NotificationCenter.default.removeObserver(token)
            engineObserver = nil
        }
        player?.stop()
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        player = nil
        engine = nil
    }

    private func fail(_ message: String) {
        dlog("ERROR: \(message)")
        status = "error: \(message)"
        // Release the file before deleting it.
        _ = writer?.finish()
        writer = nil
        if let pending = current { Recording.deleteTake(pending.take) }
        current = nil
        isRecording = false
        refreshTakes()
    }

    func refreshTakes() { takes = Recording.savedTakes() }

    func delete(_ take: String) {
        Recording.deleteTake(take)
        refreshTakes()
    }

    // MARK: route checks and notifications

    private func checkBuiltIn(_ route: AVAudioSessionRouteDescription) throws {
        let okOut: Set<AVAudioSession.Port> = [.builtInSpeaker, .builtInReceiver]
        let badOut = route.outputs.filter { !okOut.contains($0.portType) }
        let badIn = route.inputs.filter { $0.portType != .builtInMic }
        if route.outputs.isEmpty || route.inputs.isEmpty || !badOut.isEmpty || !badIn.isEmpty {
            throw RigError.notBuiltIn(AudioRig.signature(route))
        }
    }

    /// Port types + UIDs of the route; a take aborts only if this changes mid-take.
    private static func signature(_ route: AVAudioSessionRouteDescription) -> String {
        let ins = route.inputs.map { "\($0.portType.rawValue)#\($0.uid)" }.joined(separator: ",")
        let outs = route.outputs.map { "\($0.portType.rawValue)#\($0.uid)" }.joined(separator: ",")
        return "in[\(ins)] out[\(outs)]"
    }

    private func logRoute(_ route: AVAudioSessionRouteDescription) {
        for o in route.outputs {
            let names = (o.channels ?? []).map { $0.channelName }.joined(separator: ", ")
            dlog("route output: \(o.portType.rawValue) '\(o.portName)', \(o.channels?.count ?? 0) ch [\(names)]")
        }
        for i in route.inputs {
            dlog("route input: \(i.portType.rawValue) '\(i.portName)', data source \(i.selectedDataSource?.dataSourceName ?? "nil")")
        }
    }

    private func routeChanged(reason: UInt) {
        let s = AVAudioSession.sharedInstance()
        dlog("routeChangeNotification reason \(reason): \(AudioRig.signature(s.currentRoute))")
        guard isRecording, let pending = current else { return }
        let now = AudioRig.signature(s.currentRoute)
        if now != pending.routeSignature {
            abortTake("route changed mid-take (reason \(reason)) to \(now)")
        }
    }

    private func interrupted(type: UInt) {
        dlog("interruptionNotification type \(type)")
        if type == AVAudioSession.InterruptionType.began.rawValue {
            abortTake("audio session interrupted")
        }
    }
}
