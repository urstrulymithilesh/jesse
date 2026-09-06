"""Reaching this Jesse from his phone — over his own network, not through anyone else's.

Step 10. The shape: a page served from this machine, opened on his phone over
Tailscale, holding the mic open and posting raw PCM frames back. Those frames enter
the SAME `_handle_frame` path the desk microphone uses, so the real wake detector, the
real VAD and the real pipeline all run here. His replies come back as audio.

What travels: audio, in both directions, inside a WireGuard tunnel between two devices
he owns. What does not travel: anything else. No model, no memory, no transcript.

`auth` is the token layer, `transport` is the audio seam, `server` is the HTTP surface.
"""

from jesse.remote.auth import RemoteAuth, TokenError
from jesse.remote.transport import RemoteSource, SwitchingTransport

__all__ = ["RemoteAuth", "TokenError", "RemoteSource", "SwitchingTransport"]
