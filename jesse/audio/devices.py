"""Audio device enumeration + validation.

One source of truth for "which devices exist and can I actually record from this
index?", so diagnose.py, the factory, and the transport all fail the same clear way
instead of surfacing a raw PortAudioError like `Invalid device [PaErrorCode -9996]`.
"""

from __future__ import annotations

from dataclasses import dataclass


class DeviceError(RuntimeError):
    """A chosen audio device index is missing or not input-capable."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_in: int
    max_out: int
    default_samplerate: float

    @property
    def cap(self) -> str:
        if self.max_in and self.max_out:
            return "IN+OUT"
        if self.max_in:
            return "IN"
        if self.max_out:
            return "OUT"
        return "-"

    @property
    def is_input(self) -> bool:
        return self.max_in > 0


def all_devices() -> list[DeviceInfo]:
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        out.append(DeviceInfo(
            index=i, name=d["name"], host_api=hostapis[d["hostapi"]]["name"],
            max_in=d["max_input_channels"], max_out=d["max_output_channels"],
            default_samplerate=float(d["default_samplerate"]),
        ))
    return out


def input_devices() -> list[DeviceInfo]:
    return [d for d in all_devices() if d.is_input]


def default_input_index() -> int | None:
    import sounddevice as sd

    try:
        idx = sd.default.device[0]
        return idx if idx is not None and idx >= 0 else None
    except Exception:
        return None


def format_device_table(*, inputs_only: bool = False) -> str:
    devices = input_devices() if inputs_only else all_devices()
    default_in = default_input_index()
    lines = [
        f" {'idx':>3}  {'cap':<6} {'in':>2} {'out':>3}  {'host API':<22} {'defHz':>6}  name",
        " " + "-" * 92,
    ]
    for d in devices:
        mark = "  <- DEFAULT-IN" if d.index == default_in else ""
        lines.append(
            f" {d.index:>3}  {d.cap:<6} {d.max_in:>2} {d.max_out:>3}  {d.host_api:<22}"
            f" {d.default_samplerate:>6.0f}  {d.name}{mark}"
        )
    return "\n".join(lines)


def validate_input_device(index: int | None) -> None:
    """No-op for None (OS default). Otherwise raise DeviceError with a friendly,
    actionable message BEFORE any stream is opened."""
    if index is None:
        return
    devices = all_devices()
    match = next((d for d in devices if d.index == index), None)
    if match is None:
        raise DeviceError(
            f"Audio device index {index} does not exist.\n\n"
            f"Input-capable devices:\n{format_device_table(inputs_only=True)}"
        )
    if not match.is_input:
        raise DeviceError(
            f"Device {index} ('{match.name}', {match.host_api}) is OUTPUT-only "
            f"(0 input channels) — you can't record from it.\n\n"
            f"Input-capable devices:\n{format_device_table(inputs_only=True)}"
        )
