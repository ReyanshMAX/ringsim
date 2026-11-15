import SwiftUI
import UIKit

/// Single capture screen plus a Log tab (docs/CAPTURE_APP.md "UI", D-028).
struct ContentView: View {
    var body: some View {
        TabView {
            CaptureView()
                .tabItem { Label("Capture", systemImage: "waveform") }
            LogView()
                .tabItem { Label("Log", systemImage: "doc.text") }
        }
    }
}

struct CaptureView: View {
    /// UserDefaults key for the tx channel found by Protocol A step 0 (D-032).
    static let txChannelKey = "txChannel"
    /// Picker entry for a take with no playback (noise floor, D-088). Not a file name.
    static let noPlayback = "none (no playback)"
    /// Seconds between tapping Record and the take starting, so the user can step away from the rig (D-092).
    static let countdownS = 5

    @StateObject private var rig = AudioRig()

    // Defaults: bottom speaker + bottom mic (D-023).
    @State private var route: Route = .bottomSpeaker
    @State private var mic: MicSource = .bottom
    // No default channel: Record stays disabled until one is picked (Q7); the pick is stored.
    @State private var txChannel: TxChannel? =
        UserDefaults.standard.string(forKey: CaptureView.txChannelKey).flatMap(TxChannel.init(rawValue:))
    @State private var txFile: String?
    @State private var label = ""
    @State private var durationS = 10
    /// Seconds left before the take starts; nil when no countdown is running (D-092).
    @State private var countdown: Int?
    @State private var countdownTask: Task<Void, Never>?

    private let txFiles = Recording.bundledTxFiles()

    private var canRecord: Bool {
        !rig.isRecording && countdown == nil && txChannel != nil && txFile != nil && Recording.isValidLabel(label)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Output") {
                    Picker("Route", selection: $route) {
                        ForEach(Route.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Picker("tx channel", selection: $txChannel) {
                        Text("not set").tag(TxChannel?.none)
                        ForEach(TxChannel.allCases) { Text($0.rawValue).tag(TxChannel?.some($0)) }
                    }
                    .onChange(of: txChannel) { _, new in
                        if let new {
                            UserDefaults.standard.set(new.rawValue, forKey: CaptureView.txChannelKey)
                            dlog("txChannel default set to \(new.rawValue)")
                        }
                    }
                    Picker("tx file", selection: $txFile) {
                        Text("not set").tag(String?.none)
                        Text(CaptureView.noPlayback).tag(String?.some(CaptureView.noPlayback))
                        ForEach(txFiles, id: \.self) { Text($0).tag(String?.some($0)) }
                    }
                }
                Section("Input") {
                    Picker("Mic", selection: $mic) {
                        ForEach(MicSource.allCases) { Text($0.rawValue).tag($0) }
                    }
                }
                Section("Take") {
                    TextField("label [A-Za-z0-9_.-]", text: $label)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Stepper("Duration \(durationS) s", value: $durationS, in: 1...60, step: 1)
                    if let countdown {
                        Button("Starting in \(countdown) s, tap to cancel", role: .cancel) { cancelCountdown() }
                    } else {
                        Button(rig.isRecording ? "Recording..." : "Record") { startCountdown() }
                            .disabled(!canRecord)
                    }
                    Text(rig.status).font(.footnote).foregroundStyle(.secondary)
                }
                Section("Saved takes") {
                    if rig.takes.isEmpty {
                        Text("none").foregroundStyle(.secondary)
                    }
                    ForEach(rig.takes, id: \.self) { Text($0) }
                        .onDelete { idx in
                            for take in idx.map({ rig.takes[$0] }) { rig.delete(take) }
                        }
                        .deleteDisabled(rig.isRecording || countdown != nil)
                }
            }
            .navigationTitle("RingCapture")
            .onAppear { rig.refreshTakes() }
            .onDisappear { cancelCountdown() }
        }
    }

    /// Freezes the take settings at the tap, counts down `countdownS` seconds, then starts the take (D-092).
    private func startCountdown() {
        guard canRecord, let txChannel, let txFile else { return }
        // "none (no playback)" records without a player (D-088).
        let file: String? = (txFile == CaptureView.noPlayback) ? nil : txFile
        let takeLabel = label, takeRoute = route, takeMic = mic, takeDurationS = durationS
        dlog("countdown \(CaptureView.countdownS) s for take \(takeLabel)")
        countdown = CaptureView.countdownS
        countdownTask = Task { @MainActor in
            for left in stride(from: CaptureView.countdownS, to: 0, by: -1) {
                countdown = left
                do { try await Task.sleep(nanoseconds: 1_000_000_000) } catch { return }
            }
            countdown = nil
            countdownTask = nil
            rig.record(label: takeLabel, txFile: file, txChannel: txChannel, route: takeRoute,
                       mic: takeMic, durationS: takeDurationS)
        }
    }

    private func cancelCountdown() {
        guard countdown != nil else { return }
        countdownTask?.cancel()
        countdownTask = nil
        countdown = nil
        dlog("countdown cancelled")
    }
}

struct LogView: View {
    @ObservedObject private var log = DebugLog.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(log.text)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .navigationTitle("Log")
            .toolbar {
                Button("Copy") { UIPasteboard.general.string = log.text }
            }
        }
    }
}
