import SwiftUI

/// RingCapture: loops a tx WAV from `ringsim export-tx` through one iPhone speaker channel and
/// records the raw built-in mic with OS processing off (docs/CAPTURE_APP.md).
@main
struct RingCaptureApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
