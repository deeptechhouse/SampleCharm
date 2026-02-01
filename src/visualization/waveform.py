"""
Waveform visualization components following SOLID design principles.

This module provides a modular, extensible waveform display system:
- WaveformData: Immutable data model (SRP)
- WaveformRenderer: Abstract rendering interface (OCP, DIP)
- CanvasWaveformRenderer: Concrete tkinter Canvas renderer (LSP)
- WaveformController: Handles zoom/pan interactions (SRP)
- WaveformPanel: Composite widget combining all components (SRP)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np


# =============================================================================
# Data Models (Single Responsibility Principle)
# =============================================================================

@dataclass(frozen=True)
class WaveformData:
    """
    Immutable data container for waveform visualization.

    Single Responsibility: Holds and provides access to audio data for display.
    """

    audio_data: np.ndarray  # Mono audio samples (normalized -1 to 1)
    sample_rate: int
    file_name: str
    duration: float

    @classmethod
    def from_audio_sample(cls, audio_sample) -> "WaveformData":
        """
        Factory method to create WaveformData from an AudioSample.

        Args:
            audio_sample: AudioSample instance with audio_data

        Returns:
            WaveformData instance
        """
        # Get mono audio
        if audio_sample.channels == 1:
            mono_data = audio_sample.audio_data
        else:
            mono_data = np.mean(audio_sample.audio_data, axis=0)

        return cls(
            audio_data=mono_data.astype(np.float32),
            sample_rate=audio_sample.sample_rate,
            file_name=audio_sample.file_path.name,
            duration=audio_sample.duration
        )

    @classmethod
    def from_file(cls, file_path: Path, loader) -> "WaveformData":
        """
        Factory method to create WaveformData by loading a file.

        Args:
            file_path: Path to audio file
            loader: AudioLoader instance

        Returns:
            WaveformData instance
        """
        audio_sample = loader.load(file_path)
        return cls.from_audio_sample(audio_sample)

    def get_samples_in_range(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Get audio samples within a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Numpy array of samples in the range
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)

        return self.audio_data[start_sample:end_sample]

    def downsample_for_display(
        self,
        start_time: float,
        end_time: float,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample audio for efficient display, computing min/max per bucket.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            num_points: Number of display points (typically canvas width)

        Returns:
            Tuple of (min_values, max_values) arrays for envelope display
        """
        samples = self.get_samples_in_range(start_time, end_time)

        if len(samples) == 0:
            return np.zeros(num_points), np.zeros(num_points)

        if len(samples) <= num_points:
            # No downsampling needed, pad with zeros
            mins = np.zeros(num_points)
            maxs = np.zeros(num_points)
            mins[:len(samples)] = samples
            maxs[:len(samples)] = samples
            return mins, maxs

        # Compute min/max per bucket for accurate envelope
        bucket_size = len(samples) / num_points
        mins = np.zeros(num_points)
        maxs = np.zeros(num_points)

        for i in range(num_points):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size)
            if end_idx > start_idx:
                bucket = samples[start_idx:end_idx]
                mins[i] = np.min(bucket)
                maxs[i] = np.max(bucket)

        return mins, maxs


@dataclass
class ViewState:
    """
    Mutable view state for waveform display.

    Single Responsibility: Tracks current zoom/pan state.
    """

    start_time: float = 0.0  # View start in seconds
    end_time: float = 0.0    # View end in seconds
    zoom_level: float = 1.0  # 1.0 = full view, higher = zoomed in

    def get_visible_duration(self) -> float:
        """Get duration of visible portion in seconds."""
        return self.end_time - self.start_time


# =============================================================================
# Renderer Abstraction (Open/Closed Principle, Dependency Inversion)
# =============================================================================

class WaveformRenderer(ABC):
    """
    Abstract base class for waveform renderers.

    Open/Closed Principle: New rendering styles can be added without modifying
    existing code by creating new subclasses.

    Dependency Inversion: High-level WaveformPanel depends on this abstraction,
    not on concrete implementations.
    """

    @abstractmethod
    def render(
        self,
        waveform_data: WaveformData,
        view_state: ViewState,
        width: int,
        height: int
    ) -> None:
        """
        Render waveform to the display surface.

        Args:
            waveform_data: Audio data to render
            view_state: Current view state (zoom/pan)
            width: Display width in pixels
            height: Display height in pixels
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the rendering surface."""
        pass

    @abstractmethod
    def set_colors(
        self,
        waveform_color: str,
        background_color: str,
        centerline_color: str
    ) -> None:
        """
        Set rendering colors.

        Args:
            waveform_color: Color for waveform
            background_color: Background color
            centerline_color: Color for center line
        """
        pass


class CanvasWaveformRenderer(WaveformRenderer):
    """
    Tkinter Canvas-based waveform renderer.

    Liskov Substitution Principle: Can be used anywhere WaveformRenderer is expected.
    """

    def __init__(self, canvas):
        """
        Initialize renderer with a tkinter Canvas.

        Args:
            canvas: tkinter.Canvas widget to render to
        """
        self._canvas = canvas
        self._waveform_color = "#c62828"  # Red accent
        self._background_color = "#2d2d2d"
        self._centerline_color = "#404040"
        self._fill_color = "#8e1e1e"  # Dimmer red for fill

    def set_colors(
        self,
        waveform_color: str,
        background_color: str,
        centerline_color: str
    ) -> None:
        """Set rendering colors."""
        self._waveform_color = waveform_color
        self._background_color = background_color
        self._centerline_color = centerline_color
        # Derive fill color (dimmer version)
        self._fill_color = self._derive_fill_color(waveform_color)

    def _derive_fill_color(self, color: str) -> str:
        """Derive a dimmer fill color from the main color."""
        # Simple approach: return a predefined dimmer version
        # In a more complex implementation, this could parse and adjust the color
        return "#8e1e1e"

    def clear(self) -> None:
        """Clear the canvas."""
        self._canvas.delete("all")

    def render(
        self,
        waveform_data: WaveformData,
        view_state: ViewState,
        width: int,
        height: int
    ) -> None:
        """
        Render waveform to canvas.

        Uses min/max envelope rendering for efficient display of large audio files.
        """
        self.clear()

        # Draw background
        self._canvas.create_rectangle(
            0, 0, width, height,
            fill=self._background_color,
            outline=""
        )

        # Draw center line
        center_y = height // 2
        self._canvas.create_line(
            0, center_y, width, center_y,
            fill=self._centerline_color,
            width=1
        )

        if waveform_data is None:
            return

        # Get downsampled data for display
        mins, maxs = waveform_data.downsample_for_display(
            view_state.start_time,
            view_state.end_time,
            width
        )

        # Scale to canvas height (leave some padding)
        padding = 10
        usable_height = (height - 2 * padding) // 2

        # Build polygon points for filled waveform
        # Top edge (max values) left to right
        top_points = []
        for x, max_val in enumerate(maxs):
            y = center_y - int(max_val * usable_height)
            top_points.append((x, y))

        # Bottom edge (min values) right to left
        bottom_points = []
        for x, min_val in enumerate(mins):
            y = center_y - int(min_val * usable_height)
            bottom_points.append((x, y))
        bottom_points.reverse()

        # Combine into polygon
        if top_points and bottom_points:
            polygon_points = top_points + bottom_points
            flat_points = [coord for point in polygon_points for coord in point]

            if len(flat_points) >= 6:  # Need at least 3 points for polygon
                self._canvas.create_polygon(
                    flat_points,
                    fill=self._fill_color,
                    outline=self._waveform_color,
                    width=1
                )

        # Draw time markers
        self._draw_time_markers(view_state, width, height)

    def _draw_time_markers(
        self,
        view_state: ViewState,
        width: int,
        height: int
    ) -> None:
        """Draw time markers at bottom of waveform."""
        duration = view_state.end_time - view_state.start_time

        # Determine appropriate time interval for markers
        if duration <= 1:
            interval = 0.1
        elif duration <= 5:
            interval = 0.5
        elif duration <= 30:
            interval = 2
        elif duration <= 60:
            interval = 5
        else:
            interval = 10

        # Find first marker position
        start = view_state.start_time
        first_marker = (int(start / interval) + 1) * interval

        marker_y = height - 15

        current = first_marker
        while current < view_state.end_time:
            # Calculate x position
            x = int((current - start) / duration * width)

            # Draw tick
            self._canvas.create_line(
                x, height - 20, x, height - 10,
                fill=self._centerline_color,
                width=1
            )

            # Draw time label
            if current < 60:
                label = f"{current:.1f}s"
            else:
                minutes = int(current // 60)
                seconds = current % 60
                label = f"{minutes}:{seconds:04.1f}"

            self._canvas.create_text(
                x, marker_y,
                text=label,
                fill="#999999",
                font=("Helvetica", 8),
                anchor="n"
            )

            current += interval


# =============================================================================
# Controller (Single Responsibility Principle)
# =============================================================================

class WaveformController:
    """
    Handles user interactions for waveform display.

    Single Responsibility: Manages zoom, pan, and navigation interactions.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 100.0
    ZOOM_FACTOR = 1.2

    def __init__(
        self,
        on_view_change: Optional[Callable[[ViewState], None]] = None
    ):
        """
        Initialize controller.

        Args:
            on_view_change: Callback when view state changes
        """
        self._view_state = ViewState()
        self._waveform_data: Optional[WaveformData] = None
        self._on_view_change = on_view_change
        self._drag_start_x: Optional[int] = None
        self._drag_start_time: Optional[float] = None

    @property
    def view_state(self) -> ViewState:
        """Get current view state."""
        return self._view_state

    def set_waveform_data(self, data: WaveformData) -> None:
        """
        Set waveform data and reset view to full extent.

        Args:
            data: WaveformData to display
        """
        self._waveform_data = data
        self.reset_view()

    def reset_view(self) -> None:
        """Reset view to show full waveform."""
        if self._waveform_data is None:
            self._view_state.start_time = 0.0
            self._view_state.end_time = 0.0
            self._view_state.zoom_level = 1.0
        else:
            self._view_state.start_time = 0.0
            self._view_state.end_time = self._waveform_data.duration
            self._view_state.zoom_level = 1.0

        self._notify_change()

    def zoom_in(self, center_x_ratio: float = 0.5) -> None:
        """
        Zoom in on waveform.

        Args:
            center_x_ratio: Position to zoom into (0.0 = left, 1.0 = right)
        """
        self._zoom(self.ZOOM_FACTOR, center_x_ratio)

    def zoom_out(self, center_x_ratio: float = 0.5) -> None:
        """
        Zoom out on waveform.

        Args:
            center_x_ratio: Position to zoom from (0.0 = left, 1.0 = right)
        """
        self._zoom(1.0 / self.ZOOM_FACTOR, center_x_ratio)

    def _zoom(self, factor: float, center_x_ratio: float) -> None:
        """Apply zoom with given factor centered at position."""
        if self._waveform_data is None:
            return

        new_zoom = self._view_state.zoom_level * factor
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))

        if new_zoom == self._view_state.zoom_level:
            return

        # Calculate new view duration
        full_duration = self._waveform_data.duration
        new_duration = full_duration / new_zoom
        current_duration = self._view_state.end_time - self._view_state.start_time

        # Calculate center point in time
        center_time = (
            self._view_state.start_time +
            center_x_ratio * current_duration
        )

        # Calculate new start/end maintaining center point
        new_start = center_time - center_x_ratio * new_duration
        new_end = center_time + (1 - center_x_ratio) * new_duration

        # Clamp to valid range
        if new_start < 0:
            new_start = 0
            new_end = min(new_duration, full_duration)
        elif new_end > full_duration:
            new_end = full_duration
            new_start = max(0, full_duration - new_duration)

        self._view_state.start_time = new_start
        self._view_state.end_time = new_end
        self._view_state.zoom_level = new_zoom

        self._notify_change()

    def start_drag(self, x: int) -> None:
        """
        Start a drag operation.

        Args:
            x: Starting x coordinate in pixels
        """
        self._drag_start_x = x
        self._drag_start_time = self._view_state.start_time

    def update_drag(self, x: int, canvas_width: int) -> None:
        """
        Update during drag operation.

        Args:
            x: Current x coordinate
            canvas_width: Width of canvas in pixels
        """
        if (
            self._drag_start_x is None or
            self._drag_start_time is None or
            self._waveform_data is None
        ):
            return

        # Calculate time offset
        dx = self._drag_start_x - x  # Invert for natural panning
        view_duration = self._view_state.end_time - self._view_state.start_time
        time_offset = (dx / canvas_width) * view_duration

        # Calculate new start/end
        new_start = self._drag_start_time + time_offset
        new_end = new_start + view_duration

        # Clamp to valid range
        if new_start < 0:
            new_start = 0
            new_end = view_duration
        elif new_end > self._waveform_data.duration:
            new_end = self._waveform_data.duration
            new_start = max(0, new_end - view_duration)

        self._view_state.start_time = new_start
        self._view_state.end_time = new_end

        self._notify_change()

    def end_drag(self) -> None:
        """End drag operation."""
        self._drag_start_x = None
        self._drag_start_time = None

    def scroll(self, delta: float, canvas_width: int) -> None:
        """
        Scroll the view horizontally.

        Args:
            delta: Scroll amount (positive = right)
            canvas_width: Width of canvas for calculating scroll amount
        """
        if self._waveform_data is None:
            return

        view_duration = self._view_state.end_time - self._view_state.start_time
        scroll_amount = view_duration * 0.1 * delta

        new_start = self._view_state.start_time + scroll_amount
        new_end = self._view_state.end_time + scroll_amount

        # Clamp to valid range
        if new_start < 0:
            new_start = 0
            new_end = view_duration
        elif new_end > self._waveform_data.duration:
            new_end = self._waveform_data.duration
            new_start = max(0, new_end - view_duration)

        self._view_state.start_time = new_start
        self._view_state.end_time = new_end

        self._notify_change()

    def _notify_change(self) -> None:
        """Notify listeners of view state change."""
        if self._on_view_change:
            self._on_view_change(self._view_state)


# =============================================================================
# Composite Widget (Single Responsibility - combines components)
# =============================================================================

class WaveformPanel:
    """
    Complete waveform display panel widget.

    Single Responsibility: Orchestrates WaveformRenderer and WaveformController
    into a cohesive UI component.

    Dependency Inversion: Depends on WaveformRenderer abstraction, allowing
    different rendering implementations.
    """

    def __init__(
        self,
        parent,
        renderer: Optional[WaveformRenderer] = None,
        height: int = 150,
        bg_color: str = "#2d2d2d",
        waveform_color: str = "#c62828",
        centerline_color: str = "#404040"
    ):
        """
        Initialize waveform panel.

        Args:
            parent: Parent tkinter widget
            renderer: Optional custom renderer (creates CanvasWaveformRenderer if None)
            height: Panel height in pixels
            bg_color: Background color
            waveform_color: Waveform color
            centerline_color: Center line color
        """
        import tkinter as tk

        self._parent = parent
        self._height = height
        self._waveform_data: Optional[WaveformData] = None

        # Create main frame
        self._frame = tk.Frame(parent, bg=bg_color)

        # Create header with controls
        self._header = tk.Frame(self._frame, bg=bg_color)
        self._header.pack(fill=tk.X, pady=(0, 5))

        # Title label
        self._title_label = tk.Label(
            self._header,
            text="Waveform",
            bg=bg_color,
            fg="#d0d0d0",
            font=("Helvetica", 11, "normal")
        )
        self._title_label.pack(side=tk.LEFT)

        # Zoom controls
        self._controls_frame = tk.Frame(self._header, bg=bg_color)
        self._controls_frame.pack(side=tk.RIGHT)

        btn_style = {
            "bg": "#3a3a3a",
            "fg": "#f5f5f5",
            "font": ("Helvetica", 10),
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 8,
            "pady": 2,
            "cursor": "hand2",
            "activebackground": "#4a4a4a",
            "activeforeground": "#f5f5f5"
        }

        self._zoom_out_btn = tk.Button(
            self._controls_frame,
            text="-",
            command=self._on_zoom_out,
            **btn_style
        )
        self._zoom_out_btn.pack(side=tk.LEFT, padx=2)

        self._zoom_label = tk.Label(
            self._controls_frame,
            text="100%",
            bg=bg_color,
            fg="#999999",
            font=("Helvetica", 9),
            width=6
        )
        self._zoom_label.pack(side=tk.LEFT, padx=5)

        self._zoom_in_btn = tk.Button(
            self._controls_frame,
            text="+",
            command=self._on_zoom_in,
            **btn_style
        )
        self._zoom_in_btn.pack(side=tk.LEFT, padx=2)

        self._reset_btn = tk.Button(
            self._controls_frame,
            text="Reset",
            command=self._on_reset,
            **btn_style
        )
        self._reset_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Create canvas
        self._canvas = tk.Canvas(
            self._frame,
            height=height,
            bg=bg_color,
            highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Create renderer (use provided or create default)
        if renderer is not None:
            self._renderer = renderer
        else:
            self._renderer = CanvasWaveformRenderer(self._canvas)

        self._renderer.set_colors(waveform_color, bg_color, centerline_color)

        # Create controller
        self._controller = WaveformController(
            on_view_change=self._on_view_change
        )

        # Bind events
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/Mac
        self._canvas.bind("<Button-4>", self._on_scroll_up)      # Linux
        self._canvas.bind("<Button-5>", self._on_scroll_down)    # Linux

        # Info label
        self._info_label = tk.Label(
            self._frame,
            text="No audio loaded",
            bg=bg_color,
            fg="#666666",
            font=("Helvetica", 9)
        )
        self._info_label.pack(anchor=tk.W, pady=(5, 0))

    @property
    def frame(self):
        """Get the main frame widget for packing/gridding."""
        return self._frame

    def set_waveform_data(self, data: WaveformData) -> None:
        """
        Set waveform data to display.

        Args:
            data: WaveformData instance
        """
        self._waveform_data = data
        self._controller.set_waveform_data(data)
        self._update_info_label()
        self._render()

    def set_from_audio_sample(self, audio_sample) -> None:
        """
        Set waveform from an AudioSample instance.

        Args:
            audio_sample: AudioSample with audio data
        """
        data = WaveformData.from_audio_sample(audio_sample)
        self.set_waveform_data(data)

    def clear(self) -> None:
        """Clear the waveform display."""
        self._waveform_data = None
        self._controller.set_waveform_data(None)
        self._renderer.clear()
        self._info_label.config(text="No audio loaded")
        self._zoom_label.config(text="100%")

    def _render(self) -> None:
        """Render the waveform."""
        if self._waveform_data is None:
            self._renderer.clear()
            return

        width = self._canvas.winfo_width()
        height = self._canvas.winfo_height()

        if width > 1 and height > 1:  # Canvas has been sized
            self._renderer.render(
                self._waveform_data,
                self._controller.view_state,
                width,
                height
            )

    def _update_info_label(self) -> None:
        """Update the info label with current state."""
        if self._waveform_data is None:
            self._info_label.config(text="No audio loaded")
            return

        vs = self._controller.view_state
        duration = self._waveform_data.duration

        if duration < 60:
            dur_str = f"{duration:.2f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            dur_str = f"{minutes}:{seconds:05.2f}"

        info = f"{self._waveform_data.file_name} | {dur_str} | {self._waveform_data.sample_rate} Hz"

        if vs.zoom_level > 1.0:
            view_start = vs.start_time
            view_end = vs.end_time
            info += f" | View: {view_start:.2f}s - {view_end:.2f}s"

        self._info_label.config(text=info)

    def _on_view_change(self, view_state: ViewState) -> None:
        """Handle view state changes from controller."""
        self._render()
        self._update_info_label()
        zoom_pct = int(view_state.zoom_level * 100)
        self._zoom_label.config(text=f"{zoom_pct}%")

    def _on_resize(self, event) -> None:
        """Handle canvas resize."""
        self._render()

    def _on_mouse_down(self, event) -> None:
        """Handle mouse button press."""
        self._controller.start_drag(event.x)

    def _on_mouse_drag(self, event) -> None:
        """Handle mouse drag."""
        self._controller.update_drag(event.x, self._canvas.winfo_width())

    def _on_mouse_up(self, event) -> None:
        """Handle mouse button release."""
        self._controller.end_drag()

    def _on_mouse_wheel(self, event) -> None:
        """Handle mouse wheel for zooming."""
        # Calculate x ratio for zoom center
        canvas_width = self._canvas.winfo_width()
        x_ratio = event.x / canvas_width if canvas_width > 0 else 0.5

        if event.delta > 0:
            self._controller.zoom_in(x_ratio)
        else:
            self._controller.zoom_out(x_ratio)

    def _on_scroll_up(self, event) -> None:
        """Handle scroll up (Linux)."""
        canvas_width = self._canvas.winfo_width()
        x_ratio = event.x / canvas_width if canvas_width > 0 else 0.5
        self._controller.zoom_in(x_ratio)

    def _on_scroll_down(self, event) -> None:
        """Handle scroll down (Linux)."""
        canvas_width = self._canvas.winfo_width()
        x_ratio = event.x / canvas_width if canvas_width > 0 else 0.5
        self._controller.zoom_out(x_ratio)

    def _on_zoom_in(self) -> None:
        """Handle zoom in button."""
        self._controller.zoom_in()

    def _on_zoom_out(self) -> None:
        """Handle zoom out button."""
        self._controller.zoom_out()

    def _on_reset(self) -> None:
        """Handle reset button."""
        self._controller.reset_view()
