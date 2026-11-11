import Foundation

/// In-app ring-buffer log (D-028). Sideloaded builds have no debugger or console, so the
/// last 500 lines are kept in memory, shown on the Log tab and mirrored to Documents/debug.log.
/// `log` may be called from any thread; state changes happen on the main queue.
final class DebugLog: ObservableObject, @unchecked Sendable {
    static let shared = DebugLog()

    static let maxLines = 500

    @Published private(set) var lines: [String] = []

    private let fileURL: URL
    private let stamp: ISO8601DateFormatter

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = docs.appendingPathComponent("debug.log")
        stamp = ISO8601DateFormatter()
        stamp.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    }

    func log(_ message: String) {
        let line = "\(stamp.string(from: Date())) \(message)"
        DispatchQueue.main.async {
            self.lines.append(line)
            if self.lines.count > DebugLog.maxLines {
                self.lines.removeFirst(self.lines.count - DebugLog.maxLines)
            }
            self.mirror()
        }
    }

    var text: String { lines.joined(separator: "\n") }

    /// Rewrites Documents/debug.log with the current ring buffer (same 500 lines as the Log tab).
    private func mirror() {
        let data = Data((text + "\n").utf8)
        do {
            try data.write(to: fileURL, options: .atomic)
        } catch {
            // Logging the failure through `log` would recurse; keep it in memory only.
            lines.append("\(stamp.string(from: Date())) debug.log write failed: \(error.localizedDescription)")
        }
    }
}

/// Shorthand used across the app.
func dlog(_ message: String) {
    DebugLog.shared.log(message)
}
