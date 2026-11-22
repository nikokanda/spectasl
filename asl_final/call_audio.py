"""Speak to the VB-CABLE playback endpoint without changing system defaults."""
import os
import subprocess

# The speech text travels as data in the environment, never as PowerShell code.
SCRIPT = """
$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
$outputs = $voice.GetAudioOutputs()
foreach ($token in $outputs) {
    if ($token.GetDescription() -match '(?i)^CABLE Input( |$)') {
        $voice.AudioOutput = $token
        foreach ($speaker in $voice.GetVoices()) {
            if ($speaker.GetDescription() -match 'Zira') {
                $voice.Voice = $speaker
                break
            }
        }
        $voice.Rate = -1
        [void]$voice.Speak($env:ASL_SPOKEN_TEXT)
        exit 0
    }
}
throw 'CABLE Input was not found. Install VB-CABLE and restart Windows.'
"""


def speak_to_call(text):
    if os.name != "nt":
        raise RuntimeError("Call audio requires Windows and VB-CABLE.")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", SCRIPT],
        env={**os.environ, "ASL_SPOKEN_TEXT": text},
        capture_output=True, text=True, errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Windows speech output failed.")
