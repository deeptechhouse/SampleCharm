"""
SampleCharm GUI - Minimalist Audio Analysis Interface

A minimalist greyscale GUI for analyzing multiple audio files with drag-and-drop
reordering, batch processing, and configurable analysis options.
"""

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Callable
import numpy as np

# Optional PIL import for thumbnails
try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageTk = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.engine import create_analysis_engine
from src.core.loader import create_audio_loader
from src.core.models import AnalysisResult
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.visualization.waveform import WaveformPanel, WaveformData


class JackStyle:
    """Jack color scheme: Minimalist greyscale with red accents."""
    
    NAME = "Jack"
    
    # Greyscale Colors
    BG_DARK = "#1a1a1a"      # Very dark grey (almost black)
    BG_MEDIUM = "#2d2d2d"    # Dark grey
    BG_LIGHT = "#3a3a3a"     # Medium grey
    BG_LIGHTER = "#4a4a4a"   # Light grey
    BG_HOVER = "#525252"     # Hover state
    
    # Text Colors
    TEXT_PRIMARY = "#f5f5f5"   # Almost white
    TEXT_SECONDARY = "#d0d0d0" # Light grey
    TEXT_DIM = "#999999"       # Medium grey
    TEXT_DISABLED = "#666666"  # Dark grey
    
    # Red Accents (minimal use)
    ACCENT_RED = "#c62828"     # Deep red
    ACCENT_RED_LIGHT = "#e53935" # Lighter red for hover
    ACCENT_RED_DIM = "#8e1e1e"   # Dimmed red
    
    # Borders and Dividers
    BORDER = "#404040"        # Subtle grey border
    BORDER_LIGHT = "#505050"  # Lighter border
    DIVIDER = "#333333"       # Divider lines
    
    # Fonts (clean, minimal)
    FONT_TITLE = ("Helvetica", 16, "normal")
    FONT_HEADING = ("Helvetica", 11, "normal")
    FONT_BODY = ("Helvetica", 10, "normal")
    FONT_SMALL = ("Helvetica", 9, "normal")
    
    @classmethod
    def configure_widget(cls, widget, bg=None, fg=None, font=None, relief="flat", bd=0):
        """Configure widget with minimalist style."""
        style_config = {
            "bg": bg or cls.BG_DARK,
            "fg": fg or cls.TEXT_PRIMARY,
            "font": font or cls.FONT_BODY,
            "relief": relief,
            "bd": bd,
            "highlightthickness": 0,
        }
        widget.config(**style_config)


class DubStyle:
    """Dub color scheme: Green and gold with red accents."""
    
    NAME = "Dub"
    
    # Green and Gold Colors
    BG_DARK = "#0d1f0d"      # Very dark green (almost black with green tint)
    BG_MEDIUM = "#1a3a1a"    # Dark green
    BG_LIGHT = "#2d5a2d"     # Medium green
    BG_LIGHTER = "#FFCC00"   # Gold (replaces light green)
    BG_HOVER = "#FFD700"     # Brighter gold for hover state
    
    # Text Colors (black)
    TEXT_PRIMARY = "#000000"   # Black
    TEXT_SECONDARY = "#1a1a1a" # Very dark grey (almost black)
    TEXT_DIM = "#333333"       # Dark grey
    TEXT_DISABLED = "#666666"  # Medium grey
    
    # Red Accents
    ACCENT_RED = "#c62828"     # Deep red (same as Jack)
    ACCENT_RED_LIGHT = "#e53935" # Lighter red for hover
    ACCENT_RED_DIM = "#8e1e1e"   # Dimmed red
    
    # Gold Accent (for highlights)
    ACCENT_GOLD = "#FFCC00"    # Gold
    ACCENT_GOLD_LIGHT = "#FFD700" # Lighter gold
    ACCENT_GOLD_DIM = "#E6B800"   # Dimmed gold
    
    # Borders and Dividers (green-tinted)
    BORDER = "#2d4a2d"        # Green border
    BORDER_LIGHT = "#3d5a3d"  # Lighter green border
    DIVIDER = "#1a3a1a"       # Divider lines (dark green)
    
    # Fonts (same as Jack)
    FONT_TITLE = ("Helvetica", 16, "normal")
    FONT_HEADING = ("Helvetica", 11, "normal")
    FONT_BODY = ("Helvetica", 10, "normal")
    FONT_SMALL = ("Helvetica", 9, "normal")
    
    @classmethod
    def configure_widget(cls, widget, bg=None, fg=None, font=None, relief="flat", bd=0):
        """Configure widget with dub style."""
        style_config = {
            "bg": bg or cls.BG_DARK,
            "fg": fg or cls.TEXT_PRIMARY,
            "font": font or cls.FONT_BODY,
            "relief": relief,
            "bd": bd,
            "highlightthickness": 0,
        }
        widget.config(**style_config)


# Alias for backward compatibility
MinimalistStyle = JackStyle


class WaveformThumbnailGenerator:
    """
    Generates small waveform thumbnails for table display.
    
    Single Responsibility: Create thumbnail images from audio data.
    """
    
    def __init__(self, width: int = 100, height: int = 30):
        """
        Initialize thumbnail generator.
        
        Args:
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels
        """
        self.width = width
        self.height = height
    
    def generate_thumbnail(self, audio_data: np.ndarray, sample_rate: int):
        """
        Generate a thumbnail image from audio data.
        
        Args:
            audio_data: Mono audio samples (normalized -1 to 1)
            sample_rate: Sample rate of audio
            
        Returns:
            PhotoImage ready for tkinter display, or None if PIL not available
        """
        if not PIL_AVAILABLE:
            return None
        
        # Downsample audio for thumbnail
        num_samples = len(audio_data)
        target_samples = self.width
        
        if num_samples > target_samples:
            # Downsample by taking every Nth sample
            step = num_samples // target_samples
            downsampled = audio_data[::step][:target_samples]
        else:
            # Upsample by interpolation (simple repeat)
            indices = np.linspace(0, num_samples - 1, target_samples).astype(int)
            downsampled = audio_data[indices]
        
        # Create image
        img = Image.new('RGB', (self.width, self.height), color=(45, 45, 45))  # BG_MEDIUM
        draw = ImageDraw.Draw(img)
        
        # Draw center line
        center_y = self.height // 2
        draw.line([(0, center_y), (self.width, center_y)], fill=(64, 64, 64), width=1)  # BORDER
        
        # Draw waveform (min/max envelope)
        half_height = (self.height - 4) // 2  # Leave padding
        for x in range(self.width):
            if x < len(downsampled):
                # Get sample value (-1 to 1)
                sample = float(downsampled[x])
                # Scale to pixel height
                amplitude = abs(sample) * half_height
                top = center_y - amplitude
                bottom = center_y + amplitude
                # Draw line
                draw.line([(x, top), (x, bottom)], fill=(198, 40, 40), width=1)  # ACCENT_RED
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(img)


class ResultsTable:
    """
    Sortable, resizable table for displaying analysis results.
    
    Single Responsibility: Display and manage analysis results in table format.
    Open/Closed: Extensible for new columns without modifying core logic.
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_row_select: Optional[Callable[[Path], None]] = None,
        on_tag_click: Optional[Callable[[str], None]] = None,
        style: Optional[type] = None
    ):
        """
        Initialize results table.
        
        Args:
            parent: Parent widget
            on_row_select: Callback when row is selected (receives Path)
            style: MinimalistStyle class for colors
        """
        self.parent = parent
        self.on_row_select = on_row_select
        self.style = style or MinimalistStyle
        self.thumbnail_generator = WaveformThumbnailGenerator(width=100, height=30)
        
        # Store data: {path: (result, thumbnail_image)}
        self._data: Dict[Path, tuple] = {}
        # Store item IDs: {path: item_id} for tracking all items (including detached)
        self._item_map: Dict[Path, str] = {}
        self._sort_column = None
        self._sort_reverse = False
        self._tooltip_window = None
        self._filter_text = ""
        self._filter_tag = None
        
        self._build_table()
    
    def _build_table(self):
        """Build the table widget."""
        # Frame for table and scrollbars
        self.table_frame = tk.Frame(self.parent, bg=self.style.BG_DARK)
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Horizontal scrollbar
        h_scroll = tk.Scrollbar(
            self.table_frame,
            orient=tk.HORIZONTAL,
            bg=self.style.BG_DARK,
            troughcolor=self.style.BG_MEDIUM
        )
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Vertical scrollbar
        v_scroll = tk.Scrollbar(
            self.table_frame,
            orient=tk.VERTICAL,
            bg=self.style.BG_DARK,
            troughcolor=self.style.BG_MEDIUM
        )
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview (table)
        # Note: Treeview doesn't support multi-line text natively, but we'll show full content
        # with wider columns and tooltips for long text
        self.tree = ttk.Treeview(
            self.table_frame,
            columns=(
                'waveform', 'filename', 'name', 'description', 'source', 'note', 'drum',
                'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration'
            ),
            show='headings',
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
            selectmode='browse'
        )
        
        # Configure scrollbars
        h_scroll.config(command=self.tree.xview)
        v_scroll.config(command=self.tree.yview)
        
        # Pack treeview
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure columns (wider for long content like transcription and description)
        columns_config = {
            'waveform': ('Waveform', 80),
            'filename': ('Filename', 120),
            'name': ('Name', 150),
            'description': ('Description', 400),  # Wide for description with wrapping
            'source': ('Source', 80),
            'note': ('Note', 60),
            'drum': ('Drum', 60),
            'tempo': ('Tempo', 60),
            'speech': ('Speech', 60),
            'transcription': ('Transcription', 500),  # Wide for transcription with wrapping
            'tags': ('Tags', 150),
            'quality': ('Quality', 80),
            'duration': ('Duration', 60)
        }
        
        for col, (heading, width) in columns_config.items():
            self.tree.heading(col, text=heading, command=lambda c=col: self._sort_by_column(c))
            # Allow columns to expand, set minwidth for resizing
            # Make description and transcription columns wider and more scrollable
            if col in ('description', 'transcription'):
                self.tree.column(col, width=width, minwidth=150, anchor=tk.W, stretch=True)
            else:
                self.tree.column(col, width=width, minwidth=50, anchor=tk.W, stretch=True)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_selection)
        
        # Style the treeview
        self._apply_style()
        
        # Set row height - increased to accommodate wrapped text without cutting off
        # Text will be wrapped with newlines for selection/copying
        style = ttk.Style()
        style.configure('Treeview', rowheight=45)  # Increased to prevent text cutoff
        
        # Setup tag clicking after table is fully built
        self._setup_tag_clicking()
    
    def _apply_style(self):
        """Apply minimalist styling to treeview."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Treeview',
            background=self.style.BG_MEDIUM,
            foreground=self.style.TEXT_PRIMARY,
            fieldbackground=self.style.BG_MEDIUM,
            borderwidth=0,
            font=self.style.FONT_BODY
        )
        style.configure(
            'Treeview.Heading',
            background=self.style.BG_LIGHT,
            foreground=self.style.TEXT_PRIMARY,
            borderwidth=0,
            font=self.style.FONT_HEADING
        )
        style.map(
            'Treeview',
            background=[('selected', self.style.BG_LIGHTER)],
            foreground=[('selected', self.style.TEXT_PRIMARY)]
        )
        # AI feature rows get a distinct background
        self.tree.tag_configure(
            'ai_feature',
            background=self.style.BG_LIGHT,
            foreground=self.style.TEXT_DIM,
        )
    
    def _sort_by_column(self, column: str):
        """Sort table by column."""
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        
        # Get all items with their values
        items = [(self.tree.set(item, column), item) for item in self.tree.get_children('')]
        
        # Sort based on column type
        try:
            # Try numeric sort
            items.sort(key=lambda x: float(x[0]) if x[0] else 0, reverse=self._sort_reverse)
        except ValueError:
            # String sort
            items.sort(key=lambda x: x[0].lower() if x[0] else '', reverse=self._sort_reverse)
        
        # Rearrange items
        for index, (val, item) in enumerate(items):
            self.tree.move(item, '', index)
        
        # Update heading to show sort indicator
        for col in self.tree['columns']:
            heading = self.tree.heading(col)['text']
            if col == column:
                indicator = ' ▼' if self._sort_reverse else ' ▲'
                heading = heading.rstrip(' ▲▼') + indicator
            else:
                heading = heading.rstrip(' ▲▼')
            self.tree.heading(col, text=heading)
    
    def _on_selection(self, event):
        """Handle row selection."""
        selection = self.tree.selection()
        if selection and self.on_row_select:
            item = selection[0]
            # Get path from item tags or stored data
            path = self.tree.item(item, 'tags')[0] if self.tree.item(item, 'tags') else None
            if path:
                self.on_row_select(Path(path))
    
    def add_result(self, path: Path, result: AnalysisResult, audio_sample=None):
        """
        Add or update a result in the table.
        
        Args:
            path: Path to audio file
            result: Analysis result
            audio_sample: Optional AudioSample for thumbnail generation
        """
        # Generate thumbnail if audio sample available
        thumbnail = None
        if audio_sample:
            try:
                # Get mono audio
                if audio_sample.channels == 1:
                    mono_data = audio_sample.audio_data
                else:
                    mono_data = np.mean(audio_sample.audio_data, axis=0)
                
                # Normalize
                if len(mono_data) > 0:
                    max_val = np.max(np.abs(mono_data))
                    if max_val > 0:
                        mono_data = mono_data / max_val
                    
                    thumbnail = self.thumbnail_generator.generate_thumbnail(
                        mono_data, audio_sample.sample_rate
                    )
            except Exception:
                pass  # Thumbnail generation failed, continue without it
        
        # Extract values for columns
        # Get duration from audio sample if available, otherwise from quality metadata
        duration_str = ''
        if audio_sample and hasattr(audio_sample, 'duration'):
            duration_str = f"{audio_sample.duration:.1f}s"
        elif 'duration' in result.quality_metadata:
            duration_str = f"{result.quality_metadata.get('duration', 0):.1f}s"
        
        # Extract speech detection (check both contains_speech flag and transcription)
        has_speech = False
        if result.llm_analysis:
            has_speech = result.llm_analysis.contains_speech or bool(result.llm_analysis.transcription)
        
        # Extract transcription and wrap for display (Treeview doesn't support true multi-line, but we'll format it)
        transcription_text = ''
        if result.llm_analysis and result.llm_analysis.transcription:
            transcription_text = self._wrap_text_for_display(
                result.llm_analysis.transcription, 
                max_width=70  # Wider lines for transcription
            )
        
        # Get description and wrap for display
        description_text = ''
        if result.llm_analysis and result.llm_analysis.description:
            description_text = self._wrap_text_for_display(
                result.llm_analysis.description,
                max_width=60  # Wider lines for description
            )
        
        # Get all tags (no truncation, but format nicely)
        tags_text = ''
        if result.llm_analysis and result.llm_analysis.tags:
            tags_text = ', '.join(result.llm_analysis.tags)
        
        values = {
            'filename': path.name,
            'name': result.llm_analysis.suggested_name if result.llm_analysis else '',
            'description': description_text,  # Full description
            'source': result.source_classification.source_type if result.source_classification else '',
            'note': result.musical_analysis.note_name if (result.musical_analysis and result.musical_analysis.has_pitch) else '',
            'drum': result.percussive_analysis.drum_type if result.percussive_analysis else '',
            'tempo': f"{result.rhythmic_analysis.tempo_bpm:.1f}" if (result.rhythmic_analysis and result.rhythmic_analysis.has_rhythm) else '0.0',
            'speech': 'Yes' if has_speech else 'No',
            'transcription': transcription_text,  # Full transcription
            'tags': tags_text,  # All tags
            'quality': result.quality_metadata.get('quality_tier', 'Unknown'),
            'duration': duration_str
        }
        
        # Check if item exists
        existing_item = None
        for item in self.tree.get_children(''):
            item_path = self.tree.item(item, 'tags')[0] if self.tree.item(item, 'tags') else None
            if item_path == str(path):
                existing_item = item
                break
        
        # Prepare values list in column order
        # Note: waveform column shows thumbnail indicator, actual image stored separately
        # Format tags with special markers for click detection
        col_order = ['waveform', 'filename', 'name', 'description', 'source', 'note', 'drum', 'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration']
        row_values = []
        for col in col_order:
            if col == 'waveform':
                # Show indicator for waveform thumbnail
                row_values.append('●' if thumbnail else '')
            elif col == 'tags':
                # Format tags for clickability (keep original format for display)
                row_values.append(values.get(col, ''))
            else:
                row_values.append(values.get(col, ''))
        
        if existing_item:
            # Update existing item
            self.tree.item(existing_item, values=row_values)
            # Update thumbnail if available
            if thumbnail:
                self._data[path] = (result, thumbnail)
        else:
            # Insert new item
            item = self.tree.insert('', 'end', values=row_values, tags=(str(path),))
            if thumbnail:
                self._data[path] = (result, thumbnail)
            else:
                self._data[path] = (result, None)
            self._item_map[path] = item
        
        # Apply current filters
        self._apply_filters()

        # Apply current filters
        self._apply_filters()

    def add_feature_row(self, feature_name: str, summary: str, detail: str,
                        time_str: str, model: str, tag_id: str = ""):
        """
        Add an AI Feature result as a row in the table.

        Reuses existing columns:
            filename  → feature_name (e.g. "Prod. Notes")
            name      → "AI Feature"
            description → summary text (key result data)
            source    → time_str (e.g. "1.23s")
            tags      → model name
            quality   → detail (secondary info)
        Other columns are left blank.

        The row is tagged with 'ai_feature' so it can be styled differently.
        """
        col_order = [
            'waveform', 'filename', 'name', 'description', 'source',
            'note', 'drum', 'tempo', 'speech', 'transcription',
            'tags', 'quality', 'duration',
        ]
        values_map = {
            'waveform': '',
            'filename': feature_name,
            'name': 'AI Feature',
            'description': summary,
            'source': time_str,
            'note': '',
            'drum': '',
            'tempo': '',
            'speech': '',
            'transcription': detail,
            'tags': model,
            'quality': '',
            'duration': '',
        }
        row_values = [values_map.get(c, '') for c in col_order]
        self.tree.insert('', 'end', values=row_values,
                         tags=('ai_feature', tag_id))

    def clear_feature_rows(self):
        """Remove all AI Feature rows from the table, keeping analysis rows."""
        to_delete = []
        for item in self.tree.get_children(''):
            tags = self.tree.item(item, 'tags')
            if tags and 'ai_feature' in tags:
                to_delete.append(item)
        for item in to_delete:
            self.tree.delete(item)

    def clear(self):
        """Clear all rows from table."""
        for item in self.tree.get_children(''):
            self.tree.delete(item)
        self._data.clear()
        self._item_map.clear()
        self._filter_text = ""
        self._filter_tag = None
    
    def get_selected_path(self) -> Optional[Path]:
        """Get path of currently selected row."""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            path_str = self.tree.item(item, 'tags')[0] if self.tree.item(item, 'tags') else None
            return Path(path_str) if path_str else None
        return None
    
    def _setup_tag_clicking(self):
        """Setup tag clicking after table is built."""
        # Bind click events for tag clicking and cell detail viewing
        self.tree.bind('<Button-1>', self._on_tree_click)
        self.tree.bind('<Double-Button-1>', self._on_cell_double_click)
    
    def _on_tree_click(self, event):
        """Handle clicks on treeview to detect tag clicks."""
        # Get item and column under cursor
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        
        if not item or not column:
            return
        
        # Convert column identifier (#1, #2, etc.) to column name
        col_index = int(column.replace('#', '')) - 1
        columns = ['waveform', 'filename', 'name', 'description', 'source', 'note', 'drum', 'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration']
        if col_index < 0 or col_index >= len(columns):
            return
        
        col_name = columns[col_index]
        
        # If clicked on tags column, try to extract tag
        if col_name == 'tags' and self.on_tag_click:
            cell_value = self.tree.set(item, 'tags')
            if cell_value:
                # Tags are comma-separated, find which tag was clicked
                # Approximate: get x position relative to cell
                bbox = self.tree.bbox(item, column)
                if bbox:
                    cell_x, cell_y, cell_w, cell_h = bbox
                    relative_x = event.x - cell_x
                    # Simple approximation: divide tags by position
                    tags = [t.strip() for t in cell_value.split(',')]
                    if tags:
                        # Use first tag for now (could be improved with better click detection)
                        clicked_tag = tags[0] if relative_x < cell_w / 2 else (tags[-1] if len(tags) > 1 else tags[0])
                        self.on_tag_click(clicked_tag)
    
    def _copy_selected_cell(self, event):
        """Copy selected cell content to clipboard."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        # Get the column that was clicked (from last click position)
        # For now, copy all visible text from the row
        try:
            import tkinter as tk
            columns = ['waveform', 'filename', 'name', 'description', 'source', 'note', 'drum', 'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration']
            text_parts = []
            for col in ['description', 'transcription', 'name']:  # Focus on text columns
                value = self.tree.set(item, col)
                if value:
                    text_parts.append(f"{col}: {value}")
            
            if text_parts:
                text = '\n'.join(text_parts)
                self.parent.clipboard_clear()
                self.parent.clipboard_append(text)
        except Exception:
            pass
    
    def _on_cell_double_click(self, event):
        """Handle double-click on cells to show full content in detail view."""
        # Get item and column under cursor
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        
        if not item or not column:
            return
        
        # Convert column identifier (#1, #2, etc.) to column name
        col_index = int(column.replace('#', '')) - 1
        columns = ['waveform', 'filename', 'name', 'description', 'source', 'note', 'drum', 'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration']
        if col_index < 0 or col_index >= len(columns):
            return
        
        col_name = columns[col_index]
        
        # Show detail view for transcription and description
        if col_name in ('transcription', 'description'):
            path_str = self.tree.item(item, 'tags')[0] if self.tree.item(item, 'tags') else None
            if path_str and Path(path_str) in self._data:
                result, _ = self._data[Path(path_str)]
                self._show_detail_view(col_name, result, path_str)
    
    def filter_by_text(self, search_text: str):
        """Filter table by search text."""
        self._filter_text = search_text.lower()
        self._apply_filters()
    
    def filter_by_tag(self, tag: str):
        """Filter table by tag."""
        self._filter_tag = tag
        self._apply_filters()
    
    def clear_filters(self):
        """Clear all filters."""
        self._filter_text = ""
        self._filter_tag = None
        self._apply_filters()
    
    def _apply_filters(self):
        """Apply current filters to show/hide rows."""
        # Use _item_map to track all items (including detached ones)
        for path, item_id in list(self._item_map.items()):
            if path not in self._data:
                continue
            
            result, _ = self._data[path]
            
            # Check if item still exists
            try:
                # Try to get item info to verify it exists
                self.tree.item(item_id)
            except:
                # Item doesn't exist, recreate it
                result, thumbnail = self._data[path]
                self._recreate_item(path, result, thumbnail)
                item_id = self._item_map.get(path)
                if not item_id:
                    continue
            
            # Check text filter
            matches_text = True
            if self._filter_text:
                # Search across all visible columns
                try:
                    searchable_text = ' '.join([
                        str(self.tree.set(item_id, col)) for col in 
                        ['filename', 'name', 'description', 'source', 'note', 'drum', 'transcription', 'tags']
                    ]).lower()
                    matches_text = self._filter_text in searchable_text
                except:
                    matches_text = True  # If can't read, show it
            
            # Check tag filter
            matches_tag = True
            if self._filter_tag:
                if result.llm_analysis and result.llm_analysis.tags:
                    matches_tag = self._filter_tag in result.llm_analysis.tags
                else:
                    matches_tag = False
            
            # Show or hide item
            if matches_text and matches_tag:
                # Make visible - ensure it's attached
                try:
                    parent = self.tree.parent(item_id)
                    if parent != '':
                        # Reattach to root
                        self.tree.move(item_id, '', 'end')
                except:
                    # Item might not exist, recreate
                    try:
                        result, thumbnail = self._data[path]
                        self._recreate_item(path, result, thumbnail)
                    except:
                        pass
            else:
                # Hide item (detach)
                try:
                    self.tree.detach(item_id)
                except:
                    # Already detached or doesn't exist
                    pass
    
    def _wrap_text_for_display(self, text: str, max_width: int = 50) -> str:
        """Wrap text for display in Treeview cells.
        
        Note: Treeview doesn't support true multi-line text, but we can:
        1. Insert newlines to break long lines (may show as single line but helps with selection)
        2. Keep full text for selection/copying
        3. The raw output window will always show full unwrapped text
        """
        if not text:
            return ''
        
        # For very long text, wrap it to make it more readable when selected
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            # If adding this word would exceed max_width, start a new line
            if current_length + word_len + 1 > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                current_line.append(word)
                current_length += word_len + (1 if current_line else 0)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Join with newlines - Treeview may not display them as separate lines,
        # but they'll be there when text is selected/copied
        wrapped = '\n'.join(lines)
        
        # Limit total length to prevent extremely long cells (but keep full text in raw output)
        if len(wrapped) > 500:
            wrapped = wrapped[:497] + '...'
        
        return wrapped
    
    def _recreate_item(self, path: Path, result: AnalysisResult, thumbnail):
        """Recreate a table item (used when filtering)."""
        # This is a helper to recreate items - similar to add_result but without the filter check
        # Extract values (simplified version)
        duration_str = ''
        if 'duration' in result.quality_metadata:
            duration_str = f"{result.quality_metadata.get('duration', 0):.1f}s"
        
        has_speech = False
        if result.llm_analysis:
            has_speech = result.llm_analysis.contains_speech or bool(result.llm_analysis.transcription)
        
        transcription_text = ''
        if result.llm_analysis and result.llm_analysis.transcription:
            transcription_text = self._wrap_text_for_display(
                result.llm_analysis.transcription, 
                max_width=70  # Wider lines for transcription
            )
        
        description_text = ''
        if result.llm_analysis and result.llm_analysis.description:
            description_text = self._wrap_text_for_display(
                result.llm_analysis.description,
                max_width=60  # Wider lines for description
            )
        
        tags_text = ''
        if result.llm_analysis and result.llm_analysis.tags:
            tags_text = ', '.join(result.llm_analysis.tags)
        
        values = {
            'filename': path.name,
            'name': result.llm_analysis.suggested_name if result.llm_analysis else '',
            'description': description_text,
            'source': result.source_classification.source_type if result.source_classification else '',
            'note': result.musical_analysis.note_name if (result.musical_analysis and result.musical_analysis.has_pitch) else '',
            'drum': result.percussive_analysis.drum_type if result.percussive_analysis else '',
            'tempo': f"{result.rhythmic_analysis.tempo_bpm:.1f}" if (result.rhythmic_analysis and result.rhythmic_analysis.has_rhythm) else '0.0',
            'speech': 'Yes' if has_speech else 'No',
            'transcription': transcription_text,
            'tags': tags_text,
            'quality': result.quality_metadata.get('quality_tier', 'Unknown'),
            'duration': duration_str
        }
        
        col_order = ['waveform', 'filename', 'name', 'description', 'source', 'note', 'drum', 'tempo', 'speech', 'transcription', 'tags', 'quality', 'duration']
        row_values = []
        for col in col_order:
            if col == 'waveform':
                row_values.append('●' if thumbnail else '')
            else:
                row_values.append(values.get(col, ''))
        
        item = self.tree.insert('', 'end', values=row_values, tags=(str(path),))


class DraggableListbox(tk.Listbox):
    """Listbox with drag-and-drop reordering."""
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.drag_start_index = None
        self.drag_current_index = None
        
    def _on_click(self, event):
        """Handle click to start drag."""
        self.drag_start_index = self.nearest(event.y)
        
    def _on_drag(self, event):
        """Handle drag motion."""
        if self.drag_start_index is not None:
            current = self.nearest(event.y)
            if current != self.drag_start_index:
                self.drag_current_index = current
                # Visual feedback
                self.selection_clear(0, tk.END)
                self.selection_set(current)
                
    def _on_release(self, event):
        """Handle release to complete reorder."""
        if self.drag_start_index is not None:
            end_index = self.nearest(event.y)
            if end_index != self.drag_start_index and end_index >= 0:
                # Get items
                items = list(self.get(0, tk.END))
                # Move item
                item = items.pop(self.drag_start_index)
                items.insert(end_index, item)
                # Update listbox
                self.delete(0, tk.END)
                for item in items:
                    self.insert(tk.END, item)
                # Select moved item
                self.selection_set(end_index)
            self.drag_start_index = None
            self.drag_current_index = None
            
    def get_items(self) -> List[str]:
        """Get all items as list."""
        return list(self.get(0, tk.END))
    
    def set_items(self, items: List[str]):
        """Set items from list."""
        self.delete(0, tk.END)
        for item in items:
            self.insert(tk.END, item)


class SampleCharmGUI:
    """Main GUI application for SampleCharm."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SampleCharm - Audio Analysis System")
        self.root.geometry("1200x800")
        
        # Current style (default to Jack)
        self.current_style = JackStyle
        self.root.configure(bg=self.current_style.BG_DARK)
        
        # Store references to all buttons for explicit updates
        self.all_buttons = []
        
        # State
        self.file_paths: List[Path] = []
        self.results: Dict[Path, AnalysisResult] = {}
        self.audio_samples: Dict[Path, Any] = {}  # Cache loaded audio samples for waveform
        self.processing = False
        self.engine = None
        self.loader = None
        self.waveform_panel: Optional[WaveformPanel] = None
        
        # Initialize engine
        self._init_engine()
        
        # Build UI
        self._build_ui()
        
        # Apply styling
        self._apply_styling()
        
        # Ensure all buttons are styled correctly on initial load
        self._force_update_all_buttons()
        
        # Bind events to re-apply button styling after any interaction
        self._setup_button_style_protection()
        
        # Store widget references for style updates
        self._widget_refs = {}
        
    def _init_engine(self):
        """Initialize analysis engine."""
        try:
            config = load_config()
            setup_logging(level="INFO", log_format="text", colored=False, console_enabled=False)
            self.engine = create_analysis_engine(config)
            self.loader = create_audio_loader(config.get('audio', {}))
            self.feature_manager = getattr(self.engine, 'feature_manager', None)
        except Exception as e:
            self.feature_manager = None
            messagebox.showerror("Initialization Error", f"Failed to initialize engine: {e}")
            
    def _build_ui(self):
        """Build the user interface."""
        # Title bar with style toggle
        title_frame = tk.Frame(self.root, bg=self.current_style.BG_DARK, pady=20)
        title_frame.pack(fill=tk.X)
        
        # Style toggle and export (elegantly placed in top right, vertically stacked)
        toggle_frame = tk.Frame(title_frame, bg=self.current_style.BG_DARK)
        toggle_frame.pack(side=tk.RIGHT, padx=20, pady=5)
        
        # Style row
        style_row = tk.Frame(toggle_frame, bg=self.current_style.BG_DARK)
        style_row.pack(fill=tk.X, pady=(0, 5))
        
        toggle_label = tk.Label(
            style_row,
            text="Style:",
            bg=self.current_style.BG_DARK,
            fg=self.current_style.TEXT_SECONDARY,
            font=self.current_style.FONT_SMALL
        )
        toggle_label.pack(side=tk.LEFT, padx=(0, 8))
        
        # Toggle button (elegant switch)
        self.style_toggle_var = tk.StringVar(value="Jack")
        self.style_toggle = tk.Button(
            style_row,
            text="Jack",
            command=self._toggle_style,
            bg=self.current_style.BG_MEDIUM,
            fg=self.current_style.TEXT_PRIMARY,
            font=self.current_style.FONT_SMALL,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=4,
            cursor="hand2"
        )
        self.style_toggle.pack(side=tk.LEFT)
        
        # Export row (below style)
        export_row = tk.Frame(toggle_frame, bg=self.current_style.BG_DARK)
        export_row.pack(fill=tk.X)
        
        export_label = tk.Label(
            export_row,
            text="Export:",
            bg=self.current_style.BG_DARK,
            fg=self.current_style.TEXT_SECONDARY,
            font=self.current_style.FONT_SMALL
        )
        export_label.pack(side=tk.LEFT, padx=(0, 8))
        
        export_button = self._create_button(
            export_row,
            "Export",
            self._export_results,
            self.current_style.TEXT_SECONDARY,
            font=self.current_style.FONT_SMALL
        )
        export_button.pack(side=tk.LEFT)
        
        # Title (centered)
        title_label = tk.Label(
            title_frame,
            text="SampleCharm",
            bg=self.current_style.BG_DARK,
            fg=self.current_style.TEXT_PRIMARY,
            font=self.current_style.FONT_TITLE
        )
        title_label.pack(expand=True)
        
        title_label = tk.Label(
            title_frame,
            text="SampleCharm",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_TITLE,
            justify=tk.CENTER
        )
        title_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg=MinimalistStyle.BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - File management
        left_panel = tk.Frame(main_container, bg=MinimalistStyle.BG_MEDIUM, relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # File list header
        file_header = tk.Frame(left_panel, bg=MinimalistStyle.BG_MEDIUM)
        file_header.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            file_header,
            text="Files",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_HEADING
        ).pack(side=tk.LEFT)
        
        # File list with scrollbar
        list_frame = tk.Frame(left_panel, bg=MinimalistStyle.BG_DARK)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame, bg=MinimalistStyle.BG_DARK, troughcolor=MinimalistStyle.BG_MEDIUM)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = DraggableListbox(
            list_frame,
            bg=MinimalistStyle.BG_MEDIUM,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectbackground=MinimalistStyle.BG_LIGHTER,
            selectforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            selectmode=tk.EXTENDED  # Enable multiple selection
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # Bind selection event to show waveform and update button
        def on_selection_change(event):
            self._on_file_select(event)
            self._update_analyze_button()
        
        self.file_listbox.bind("<<ListboxSelect>>", on_selection_change)
        
        # File buttons (minimal, horizontal)
        file_buttons = tk.Frame(left_panel, bg=MinimalistStyle.BG_DARK)
        file_buttons.pack(fill=tk.X, pady=(0, 20))
        
        self._create_button(
            file_buttons,
            "Load Files",
            self._load_files,
            self.current_style.TEXT_PRIMARY
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._create_button(
            file_buttons,
            "Remove",
            self._remove_selected,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._create_button(
            file_buttons,
            "Select All",
            self._select_all,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._create_button(
            file_buttons,
            "Deselect All",
            self._deselect_all,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._create_button(
            file_buttons,
            "Clear",
            self._clear_all,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT)
        
        # Options panel (minimal)
        options_frame = tk.Frame(left_panel, bg=MinimalistStyle.BG_DARK)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Divider
        divider1 = tk.Frame(left_panel, bg=MinimalistStyle.DIVIDER, height=1)
        divider1.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            options_frame,
            text="Options",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_HEADING
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # LLM Description toggle
        self.llm_enabled = tk.BooleanVar(value=True)
        llm_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_MEDIUM)
        llm_frame.pack(fill=tk.X, padx=5, pady=2)

        llm_check = tk.Checkbutton(
            llm_frame,
            text="Enable LLM Descriptions",
            variable=self.llm_enabled,
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectcolor=MinimalistStyle.BG_MEDIUM,
            activebackground=MinimalistStyle.BG_DARK,
            activeforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            relief=tk.FLAT
        )
        llm_check.pack(side=tk.LEFT)

        # Speech Recognition toggle (Whisper - can be slow on large files)
        self.speech_enabled = tk.BooleanVar(value=True)
        speech_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_MEDIUM)
        speech_frame.pack(fill=tk.X, padx=5, pady=2)

        speech_check = tk.Checkbutton(
            speech_frame,
            text="Enable Speech Recognition",
            variable=self.speech_enabled,
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectcolor=MinimalistStyle.BG_MEDIUM,
            activebackground=MinimalistStyle.BG_DARK,
            activeforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            relief=tk.FLAT
        )
        speech_check.pack(side=tk.LEFT)
        
        # Skip large files toggle
        self.skip_large = tk.BooleanVar(value=False)
        skip_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_MEDIUM)
        skip_frame.pack(fill=tk.X, padx=5, pady=2)
        
        skip_check = tk.Checkbutton(
            skip_frame,
            text="Skip files > 10MB",
            variable=self.skip_large,
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectcolor=MinimalistStyle.BG_MEDIUM,
            activebackground=MinimalistStyle.BG_DARK,
            activeforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            relief=tk.FLAT
        )
        skip_check.pack(side=tk.LEFT)

        # AI Features section (collapsible, grouped 2-column grid)
        ai_divider = tk.Frame(options_frame, bg=MinimalistStyle.DIVIDER, height=1)
        ai_divider.pack(fill=tk.X, pady=(10, 5))

        self.ai_section_visible = tk.BooleanVar(value=False)
        ai_header_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_DARK)
        ai_header_frame.pack(fill=tk.X, pady=(0, 5))

        self.ai_toggle_label = tk.Label(
            ai_header_frame,
            text="\u25b6 AI Features (10/10)",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_HEADING,
            cursor="hand2"
        )
        self.ai_toggle_label.pack(anchor=tk.W)
        self.ai_toggle_label.bind("<Button-1>", lambda e: self._toggle_ai_section())

        self.ai_features_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_DARK)
        # Initially hidden (not packed)

        # Feature groups: (feature_id, short_label)
        AI_FEATURE_GROUPS = [
            ("Per-Sample", [
                ("production_notes", "Prod. Notes"),
                ("speech_deep_analyzer", "Speech Analyzer"),
                ("natural_language_search", "NL Search"),
                ("similar_sample_finder", "Similar Finder"),
            ]),
            ("Batch", [
                ("sample_pack_curator", "Pack Curator"),
                ("batch_rename", "Batch Rename"),
                ("daw_suggestions", "DAW Suggest."),
                ("sample_chain", "Sample Chain"),
                ("marketplace_description", "Marketplace"),
                ("anomaly_reporter", "Anomaly Report"),
            ]),
        ]

        # Store for display name lookup and checkbox dim/undim
        self.ai_feature_vars: Dict[str, tk.BooleanVar] = {}
        self.ai_feature_checks: Dict[str, tk.Checkbutton] = {}
        self.ai_feature_names: Dict[str, str] = {}

        for group_name, features in AI_FEATURE_GROUPS:
            # Group label
            group_label = tk.Label(
                self.ai_features_frame,
                text=group_name,
                bg=MinimalistStyle.BG_DARK,
                fg=MinimalistStyle.TEXT_DIM,
                font=MinimalistStyle.FONT_SMALL,
            )
            group_label.pack(anchor=tk.W, padx=5, pady=(4, 1))

            # 2-column grid for this group
            grid_frame = tk.Frame(self.ai_features_frame, bg=MinimalistStyle.BG_DARK)
            grid_frame.pack(fill=tk.X, padx=5)
            grid_frame.columnconfigure(0, weight=1)
            grid_frame.columnconfigure(1, weight=1)

            for idx, (feature_id, short_label) in enumerate(features):
                var = tk.BooleanVar(value=True)
                self.ai_feature_vars[feature_id] = var
                self.ai_feature_names[feature_id] = short_label

                row = idx // 2
                col = idx % 2

                cb = tk.Checkbutton(
                    grid_frame,
                    text=short_label,
                    variable=var,
                    bg=MinimalistStyle.BG_DARK,
                    fg=MinimalistStyle.TEXT_PRIMARY,
                    selectcolor=MinimalistStyle.BG_MEDIUM,
                    activebackground=MinimalistStyle.BG_DARK,
                    activeforeground=MinimalistStyle.TEXT_PRIMARY,
                    font=MinimalistStyle.FONT_SMALL,
                    relief=tk.FLAT,
                    command=lambda fid=feature_id: self._sync_feature_toggle(fid),
                )
                cb.grid(row=row, column=col, sticky=tk.W, pady=0)
                self.ai_feature_checks[feature_id] = cb

        # --- Feature-specific parameter inputs ---
        params_frame = tk.Frame(self.ai_features_frame, bg=MinimalistStyle.BG_DARK)
        params_frame.pack(fill=tk.X, padx=5, pady=(6, 0))

        param_label_cfg = dict(
            bg=MinimalistStyle.BG_DARK, fg=MinimalistStyle.TEXT_DIM,
            font=MinimalistStyle.FONT_SMALL,
        )
        param_entry_cfg = dict(
            bg=MinimalistStyle.BG_MEDIUM, fg=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_SMALL, relief=tk.FLAT,
            insertbackground=MinimalistStyle.TEXT_PRIMARY,
        )

        # NL Search query
        row_nl = tk.Frame(params_frame, bg=MinimalistStyle.BG_DARK)
        row_nl.pack(fill=tk.X, pady=1)
        tk.Label(row_nl, text="NL Query:", width=10, anchor=tk.W,
                 **param_label_cfg).pack(side=tk.LEFT)
        self.ai_nl_query = tk.Entry(row_nl, width=20, **param_entry_cfg)
        self.ai_nl_query.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # DAW context (BPM / Key / Genre)
        row_daw = tk.Frame(params_frame, bg=MinimalistStyle.BG_DARK)
        row_daw.pack(fill=tk.X, pady=1)
        tk.Label(row_daw, text="DAW BPM:", width=10, anchor=tk.W,
                 **param_label_cfg).pack(side=tk.LEFT)
        self.ai_daw_bpm = tk.Entry(row_daw, width=5, **param_entry_cfg)
        self.ai_daw_bpm.insert(0, "120")
        self.ai_daw_bpm.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(row_daw, text="Key:", **param_label_cfg).pack(side=tk.LEFT)
        self.ai_daw_key = tk.Entry(row_daw, width=7, **param_entry_cfg)
        self.ai_daw_key.insert(0, "C major")
        self.ai_daw_key.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(row_daw, text="Genre:", **param_label_cfg).pack(side=tk.LEFT)
        self.ai_daw_genre = tk.Entry(row_daw, width=8, **param_entry_cfg)
        self.ai_daw_genre.insert(0, "electronic")
        self.ai_daw_genre.pack(side=tk.LEFT)

        # Marketplace pack name
        row_mkt = tk.Frame(params_frame, bg=MinimalistStyle.BG_DARK)
        row_mkt.pack(fill=tk.X, pady=1)
        tk.Label(row_mkt, text="Pack Name:", width=10, anchor=tk.W,
                 **param_label_cfg).pack(side=tk.LEFT)
        self.ai_pack_name = tk.Entry(row_mkt, width=20, **param_entry_cfg)
        self.ai_pack_name.insert(0, "Untitled Pack")
        self.ai_pack_name.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Sample Chain energy preference
        row_chain = tk.Frame(params_frame, bg=MinimalistStyle.BG_DARK)
        row_chain.pack(fill=tk.X, pady=1)
        tk.Label(row_chain, text="Energy:", width=10, anchor=tk.W,
                 **param_label_cfg).pack(side=tk.LEFT)
        self.ai_energy_pref = tk.StringVar(value="ascending")
        for val in ("ascending", "descending", "arc", "flat"):
            tk.Radiobutton(
                row_chain, text=val.capitalize(), variable=self.ai_energy_pref,
                value=val, bg=MinimalistStyle.BG_DARK,
                fg=MinimalistStyle.TEXT_DIM, selectcolor=MinimalistStyle.BG_MEDIUM,
                activebackground=MinimalistStyle.BG_DARK,
                activeforeground=MinimalistStyle.TEXT_PRIMARY,
                font=MinimalistStyle.FONT_SMALL, relief=tk.FLAT,
            ).pack(side=tk.LEFT)

        # Run AI Features button (initially hidden — shown when features enabled + results exist)
        self.ai_run_btn_frame = tk.Frame(self.ai_features_frame, bg=MinimalistStyle.BG_DARK)
        self._create_button(
            self.ai_run_btn_frame,
            "Run AI Features",
            self._run_ai_features,
            MinimalistStyle.TEXT_SECONDARY
        ).pack(side=tk.LEFT)
        # Button visibility managed by _update_run_button_visibility()

        # Processing mode
        mode_frame = tk.Frame(options_frame, bg=MinimalistStyle.BG_DARK)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            mode_frame,
            text="Mode:",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_BODY
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.processing_mode = tk.StringVar(value="batch")
        tk.Radiobutton(
            mode_frame,
            text="Batch",
            variable=self.processing_mode,
            value="batch",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectcolor=MinimalistStyle.BG_MEDIUM,
            activebackground=MinimalistStyle.BG_DARK,
            activeforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Radiobutton(
            mode_frame,
            text="Sequential",
            variable=self.processing_mode,
            value="sequential",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            selectcolor=MinimalistStyle.BG_MEDIUM,
            activebackground=MinimalistStyle.BG_DARK,
            activeforeground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            highlightthickness=0
        ).pack(side=tk.LEFT)
        
        # Divider
        divider2 = tk.Frame(left_panel, bg=MinimalistStyle.DIVIDER, height=1)
        divider2.pack(fill=tk.X, pady=(0, 15))
        
        # Control buttons (minimal, red accent for primary action)
        control_frame = tk.Frame(left_panel, bg=MinimalistStyle.BG_DARK)
        control_frame.pack(fill=tk.X)
        
        self.analyze_button = self._create_button(
            control_frame,
            "Analyze All",
            self._start_analysis,
            self.current_style.ACCENT_RED,
            font=self.current_style.FONT_HEADING
        )
        self.analyze_button.pack(fill=tk.X, pady=(0, 8))
        
        self.stop_button = self._create_button(
            control_frame,
            "Stop",
            self._stop_analysis,
            self.current_style.TEXT_SECONDARY,
            state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X)
        
        # Right panel - Results
        right_panel = tk.Frame(main_container, bg=MinimalistStyle.BG_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Results header (minimal)
        results_header = tk.Frame(right_panel, bg=MinimalistStyle.BG_DARK)
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            results_header,
            text="Results",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_HEADING
        ).pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            right_panel,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress label
        self.progress_label = tk.Label(
            right_panel,
            text="Ready",
            bg=MinimalistStyle.BG_MEDIUM,
            fg=MinimalistStyle.TEXT_SECONDARY,
            font=MinimalistStyle.FONT_SMALL
        )
        self.progress_label.pack(anchor=tk.W, padx=5, pady=2)

        # Waveform display panel
        self.waveform_panel = WaveformPanel(
            right_panel,
            height=150,
            bg_color=self.current_style.BG_MEDIUM,
            waveform_color=self.current_style.ACCENT_RED,
            centerline_color=self.current_style.BORDER
        )
        self.waveform_panel.frame.pack(fill=tk.X, padx=5, pady=(10, 10))
        
        # Store waveform panel buttons in all_buttons for explicit updates
        if hasattr(self.waveform_panel, '_zoom_in_btn'):
            # Add waveform panel buttons to tracked buttons list
            self.all_buttons.append(self.waveform_panel._zoom_in_btn)
            self.all_buttons.append(self.waveform_panel._zoom_out_btn)
            self.all_buttons.append(self.waveform_panel._reset_btn)
            
            # Update waveform panel buttons immediately with current style (full config)
            if self.current_style.NAME == "Dub":
                for btn in [self.waveform_panel._zoom_in_btn, 
                           self.waveform_panel._zoom_out_btn, 
                           self.waveform_panel._reset_btn]:
                    # Apply multiple times to override hardcoded styles
                    for _ in range(5):
                        btn.config(
                            bg="#FFCC00",
                            fg="#0d1f0d",
                            activebackground="#FFD700",
                            activeforeground="#0d1f0d",
                            highlightbackground="#FFCC00",
                            highlightcolor="#FFCC00",
                            relief=tk.FLAT,
                            borderwidth=0
                        )
                        btn.configure(
                            bg="#FFCC00",
                            fg="#0d1f0d",
                            activebackground="#FFD700",
                            activeforeground="#0d1f0d",
                            highlightbackground="#FFCC00",
                            highlightcolor="#FFCC00",
                            relief=tk.FLAT,
                            borderwidth=0
                        )
            else:
                for btn in [self.waveform_panel._zoom_in_btn, 
                           self.waveform_panel._zoom_out_btn, 
                           self.waveform_panel._reset_btn]:
                    # Apply multiple times to override hardcoded styles
                    for _ in range(5):
                        btn.config(
                            bg="#4a4a4a",
                            fg="#000000",
                            activebackground="#525252",
                            activeforeground="#000000",
                            highlightbackground="#4a4a4a",
                            highlightcolor="#4a4a4a",
                            relief=tk.FLAT,
                            borderwidth=0
                        )
                        btn.configure(
                            bg="#4a4a4a",
                            fg="#000000",
                            activebackground="#525252",
                            activeforeground="#000000",
                            highlightbackground="#4a4a4a",
                            highlightcolor="#4a4a4a",
                            relief=tk.FLAT,
                            borderwidth=0
                        )

        # Search/filter frame
        search_frame = tk.Frame(right_panel, bg=MinimalistStyle.BG_DARK)
        search_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        search_label = tk.Label(
            search_frame,
            text="Search:",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY
        )
        search_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._on_search_change)
        search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            bg=MinimalistStyle.BG_MEDIUM,
            fg=MinimalistStyle.TEXT_PRIMARY,
            insertbackground=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            relief=tk.FLAT,
            borderwidth=1
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        clear_search_btn = self._create_button(
            search_frame,
            "Clear",
            lambda: self.search_var.set(""),
            MinimalistStyle.TEXT_SECONDARY
        )
        clear_search_btn.pack(side=tk.LEFT)
        
        # Results table (reduced space to make room for output window)
        results_table_frame = tk.Frame(right_panel, bg=MinimalistStyle.BG_DARK)
        results_table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create inner frame for table (below hint)
        table_inner_frame = tk.Frame(results_table_frame, bg=MinimalistStyle.BG_DARK)
        table_inner_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_table = ResultsTable(
            table_inner_frame,
            on_row_select=self._on_table_row_select,
            on_tag_click=self._on_tag_click,
            style=self.current_style
        )
        
        # Results buttons (minimal)
        results_buttons = tk.Frame(right_panel, bg=MinimalistStyle.BG_DARK)
        results_buttons.pack(fill=tk.X)
        
        self._create_button(
            results_buttons,
            "Export",
            self._export_results,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._create_button(
            results_buttons,
            "Clear",
            self._clear_results,
            self.current_style.TEXT_SECONDARY
        ).pack(side=tk.LEFT)
        
        # Add hint label above table
        table_hint = tk.Label(
            results_table_frame,
            text="💡 Double-click Transcription or Description cells to view full text | Full content shown in Raw Output below",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.ACCENT_RED,  # Make it more visible
            font=MinimalistStyle.FONT_SMALL
        )
        table_hint.pack(anchor=tk.W, padx=5, pady=(0, 5))
        
        # Raw output window at bottom (larger, more prominent)
        # Add divider to separate from table
        output_divider = tk.Frame(right_panel, bg=MinimalistStyle.DIVIDER, height=2)
        output_divider.pack(fill=tk.X, pady=(10, 5))
        
        output_frame = tk.Frame(right_panel, bg=MinimalistStyle.BG_DARK)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))  # Takes remaining space
        
        # Header with prominent label
        output_header = tk.Frame(output_frame, bg=MinimalistStyle.BG_DARK)
        output_header.pack(fill=tk.X, padx=5, pady=(5, 5))
        
        output_label = tk.Label(
            output_header,
            text="📄 Raw Output - Full Transcription & Description (Scroll to see all content):",
            bg=MinimalistStyle.BG_DARK,
            fg=MinimalistStyle.ACCENT_RED,  # Make it stand out
            font=MinimalistStyle.FONT_HEADING
        )
        output_label.pack(side=tk.LEFT)
        
        # Text widget with scrollbar for raw output
        output_text_frame = tk.Frame(output_frame, bg=MinimalistStyle.BG_DARK)
        output_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        output_v_scroll = tk.Scrollbar(
            output_text_frame,
            orient=tk.VERTICAL,
            bg=MinimalistStyle.BG_DARK,
            troughcolor=MinimalistStyle.BG_MEDIUM
        )
        output_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.output_text = tk.Text(
            output_text_frame,
            height=15,  # Increased height for better visibility
            bg=MinimalistStyle.BG_MEDIUM,
            fg=MinimalistStyle.TEXT_PRIMARY,
            font=MinimalistStyle.FONT_BODY,
            relief=tk.FLAT,
            borderwidth=1,
            wrap=tk.WORD,
            yscrollcommand=output_v_scroll.set
        )
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_v_scroll.config(command=self.output_text.yview)
        
    def _toggle_style(self):
        """Toggle between Jack and Dub color schemes."""
        if self.current_style == JackStyle:
            self.current_style = DubStyle
            self.style_toggle_var.set("Dub")
        else:
            self.current_style = JackStyle
            self.style_toggle_var.set("Jack")
        
        # Update toggle button text and style
        if self.current_style.NAME == "Dub":
            self.style_toggle.config(
                text=self.current_style.NAME,
                bg="#FFCC00",  # Gold
                fg="#0d1f0d"   # Green text
            )
        else:
            self.style_toggle.config(
                text=self.current_style.NAME,
                bg="#4a4a4a",  # Light grey
                fg="#000000"   # Black text
            )
        
        # Update all widgets with new style
        self._update_style()
        
        # Force immediate GUI update - multiple passes to clear cache
        for _ in range(3):
            self.root.update_idletasks()
            self.root.update()
        
        # Aggressively update all buttons multiple times (macOS needs this)
        # Immediate update first
        self._force_update_all_buttons()
        # Then scheduled updates with cache clearing
        for delay in [10, 50, 100, 200, 500, 1000]:
            self.root.after(delay, lambda: self._force_update_all_buttons() or self.root.update_idletasks() or self.root.update())
    
    def _setup_button_style_protection(self):
        """Set up event bindings to protect button styling."""
        def protect_buttons_after_event(event=None):
            """Re-apply button styling after any event."""
            # Use after() to avoid recursion and ensure it happens after the event
            self.root.after(10, self._force_update_all_buttons)
        
        # Bind to common events that might reset button styling
        self.root.bind('<Button-1>', protect_buttons_after_event)
        self.root.bind('<ButtonRelease-1>', protect_buttons_after_event)
        self.root.bind('<FocusIn>', protect_buttons_after_event)
        self.root.bind('<FocusOut>', protect_buttons_after_event)
        
        # Also bind to all buttons directly
        def bind_to_button(btn):
            """Recursively bind events to all buttons."""
            try:
                if btn.winfo_class() == 'Button':
                    try:
                        text = btn.cget('text')
                        if text not in ['Jack', 'Dub']:
                            btn.bind('<Button-1>', protect_buttons_after_event)
                            btn.bind('<ButtonRelease-1>', protect_buttons_after_event)
                            btn.bind('<Enter>', protect_buttons_after_event)
                            btn.bind('<Leave>', protect_buttons_after_event)
                    except:
                        pass
            except:
                pass
            
            try:
                for child in btn.winfo_children():
                    bind_to_button(child)
            except:
                pass
        
        # Bind to all existing buttons
        bind_to_button(self.root)
        
        # Also set up a periodic refresh for buttons (every 500ms)
        def periodic_button_refresh():
            if self.current_style.NAME == "Dub":
                self._force_update_all_buttons()
            self.root.after(500, periodic_button_refresh)
        
        periodic_button_refresh()
    
    def _reapply_button_style(self, btn):
        """Re-apply current style to a single button."""
        if not btn or btn.winfo_class() != 'Button':
            return
        try:
            text = btn.cget('text')
            if text in ['Jack', 'Dub']:
                return  # Skip style toggle
        except:
            pass
        
        if self.current_style.NAME == "Dub":
            for _ in range(5):  # More aggressive
                btn.config(
                    bg="#FFCC00",
                    fg="#0d1f0d",
                    activebackground="#FFD700",
                    activeforeground="#0d1f0d",
                    highlightbackground="#FFCC00",
                    highlightcolor="#FFCC00",
                    relief=tk.FLAT,
                    borderwidth=0,
                    overrelief=tk.FLAT,
                    highlightthickness=0
                )
                btn.configure(
                    bg="#FFCC00",
                    fg="#0d1f0d",
                    activebackground="#FFD700",
                    activeforeground="#0d1f0d",
                    highlightbackground="#FFCC00",
                    highlightcolor="#FFCC00",
                    relief=tk.FLAT,
                    borderwidth=0,
                    overrelief=tk.FLAT,
                    highlightthickness=0
                )
            # Force a widget update
            try:
                btn.update_idletasks()
            except:
                pass
        else:
            for _ in range(3):
                btn.config(
                    bg="#4a4a4a",
                    fg="#000000",
                    activebackground="#525252",
                    activeforeground="#000000",
                    highlightbackground="#4a4a4a",
                    highlightcolor="#4a4a4a",
                    relief=tk.FLAT,
                    borderwidth=0
                )
                btn.configure(
                    bg="#4a4a4a",
                    fg="#000000",
                    activebackground="#525252",
                    activeforeground="#000000",
                    highlightbackground="#4a4a4a",
                    highlightcolor="#4a4a4a",
                    relief=tk.FLAT,
                    borderwidth=0
                )
    
    def _force_update_all_buttons(self):
        """Force update all buttons - more aggressive version matching stop button styling."""
        def force_button_update(widget):
            """Recursively force update all buttons with full configuration."""
            try:
                if widget.winfo_class() == 'Button':
                    try:
                        text = widget.cget('text')
                        if text not in ['Jack', 'Dub']:
                            # Use exact same configuration as _create_button uses
                            if self.current_style.NAME == "Dub":
                                # Match stop button: gold background, green text, all options
                                # Apply multiple times to override system styling
                                for _ in range(3):  # Try 3 times
                                    widget.config(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0,
                                    overrelief=tk.FLAT,
                                    highlightthickness=0
                                )
                                widget.configure(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0,
                                    overrelief=tk.FLAT,
                                    highlightthickness=0
                                )
                                # Force widget update
                                try:
                                    widget.update_idletasks()
                                except:
                                    pass
                            else:
                                # Jack style: grey background, black text
                                for _ in range(3):  # Try 3 times
                                    widget.config(
                                        bg="#4a4a4a",
                                        fg="#000000",
                                        activebackground="#525252",
                                        activeforeground="#000000",
                                        highlightbackground="#4a4a4a",
                                        highlightcolor="#4a4a4a",
                                        relief=tk.FLAT,
                                        borderwidth=0,
                                        overrelief=tk.FLAT,
                                        highlightthickness=0
                                    )
                                    widget.configure(
                                        bg="#4a4a4a",
                                        fg="#000000",
                                        activebackground="#525252",
                                        activeforeground="#000000",
                                        highlightbackground="#4a4a4a",
                                        highlightcolor="#4a4a4a",
                                        relief=tk.FLAT,
                                        borderwidth=0,
                                        overrelief=tk.FLAT,
                                        highlightthickness=0
                                    )
                                # Force widget update
                                try:
                                    widget.update_idletasks()
                                except:
                                    pass
                    except:
                        pass
            except:
                pass
            
            try:
                for child in widget.winfo_children():
                    force_button_update(child)
            except:
                pass
        
        # Run multiple passes
        for _ in range(3):
            force_button_update(self.root)
        
        # Explicitly update all stored button references (like stop button)
        if hasattr(self, 'all_buttons'):
            for btn in self.all_buttons:
                try:
                    text = btn.cget('text')
                    if text not in ['Jack', 'Dub']:
                        if self.current_style.NAME == "Dub":
                            # Match stop button exactly - apply multiple times to override cache
                            for _ in range(5):  # Apply 5 times to force override
                                btn.config(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                                btn.configure(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                            # Force widget to update
                            btn.update_idletasks()
                        else:
                            # Jack style
                            for _ in range(5):  # Apply 5 times to force override
                                btn.config(
                                    bg="#4a4a4a",
                                    fg="#000000",
                                    activebackground="#525252",
                                    activeforeground="#000000",
                                    highlightbackground="#4a4a4a",
                                    highlightcolor="#4a4a4a",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                                btn.configure(
                                    bg="#4a4a4a",
                                    fg="#000000",
                                    activebackground="#525252",
                                    activeforeground="#000000",
                                    highlightbackground="#4a4a4a",
                                    highlightcolor="#4a4a4a",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                            # Force widget to update
                            btn.update_idletasks()
                except:
                    pass
        
        # Also update waveform panel buttons explicitly with full configuration
        if hasattr(self, 'waveform_panel') and self.waveform_panel:
            try:
                if hasattr(self.waveform_panel, '_zoom_in_btn'):
                    if self.current_style.NAME == "Dub":
                        # Match stop button styling exactly - apply multiple times
                        for btn in [self.waveform_panel._zoom_in_btn, 
                                   self.waveform_panel._zoom_out_btn, 
                                   self.waveform_panel._reset_btn]:
                            # Apply 5 times to override hardcoded styles
                            for _ in range(5):
                                btn.config(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                                btn.configure(
                                    bg="#FFCC00",
                                    fg="#0d1f0d",
                                    activebackground="#FFD700",
                                    activeforeground="#0d1f0d",
                                    highlightbackground="#FFCC00",
                                    highlightcolor="#FFCC00",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                    else:
                        # Jack style
                        for btn in [self.waveform_panel._zoom_in_btn, 
                                   self.waveform_panel._zoom_out_btn, 
                                   self.waveform_panel._reset_btn]:
                            # Apply 5 times to override hardcoded styles
                            for _ in range(5):
                                btn.config(
                                    bg="#4a4a4a",
                                    fg="#000000",
                                    activebackground="#525252",
                                    activeforeground="#000000",
                                    highlightbackground="#4a4a4a",
                                    highlightcolor="#4a4a4a",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
                                btn.configure(
                                    bg="#4a4a4a",
                                    fg="#000000",
                                    activebackground="#525252",
                                    activeforeground="#000000",
                                    highlightbackground="#4a4a4a",
                                    highlightcolor="#4a4a4a",
                                    relief=tk.FLAT,
                                    borderwidth=0
                                )
            except:
                pass
        
        self.root.update_idletasks()
    
    def _update_style(self):
        """Update all widgets to use current style."""
        # Update root window
        self.root.configure(bg=self.current_style.BG_DARK)
        
        # Explicitly update all buttons FIRST (most important)
        self._update_all_buttons()
        
        # Then update all frames and widgets recursively
        self._update_widget_style(self.root)
        
        # Force another button update after widget style update (in case anything was reset)
        self._update_all_buttons()
        
        # Also call the aggressive force update
        self._force_update_all_buttons()
        
        # Update ttk styles
        style = ttk.Style()
        style.configure(
            'TProgressbar',
            background=self.current_style.ACCENT_RED,
            troughcolor=self.current_style.BG_MEDIUM,
            borderwidth=0,
            lightcolor=self.current_style.ACCENT_RED,
            darkcolor=self.current_style.ACCENT_RED
        )
        
        # Update results table if it exists
        if hasattr(self, 'results_table') and self.results_table:
            self.results_table.style = self.current_style
            # Rebuild table styles
            self.results_table._apply_style()
            # Update treeview colors - use black text for Dub style
            text_color = "#000000" if self.current_style.NAME == "Dub" else self.current_style.TEXT_PRIMARY
            style = ttk.Style()
            style.configure(
                'Treeview',
                background=self.current_style.BG_MEDIUM,
                foreground=text_color,
                fieldbackground=self.current_style.BG_MEDIUM,
                borderwidth=0,
                font=self.current_style.FONT_BODY
            )
            style.configure(
                'Treeview.Heading',
                background=self.current_style.BG_LIGHT,
                foreground=text_color,
                borderwidth=0,
                font=self.current_style.FONT_HEADING
            )
            style.map('Treeview',
                      background=[('selected', self.current_style.BG_LIGHTER)],
                      foreground=[('selected', text_color)])
        
        # Update waveform panel if it exists
        if hasattr(self, 'waveform_panel') and self.waveform_panel:
            # Waveform panel uses its own colors, but we can update if needed
            pass
    
    def _update_widget_style(self, widget):
        """Recursively update widget styles."""
        try:
            widget_type = widget.winfo_class()
            
            # Update frames
            if widget_type == 'Frame':
                if hasattr(widget, 'cget') and widget.cget('bg'):
                    widget.configure(bg=self.current_style.BG_DARK)
            
            # Update labels
            elif widget_type == 'Label':
                if hasattr(widget, 'cget'):
                    current_bg = widget.cget('bg')
                    current_fg = widget.cget('fg')
                    # Only update if it's a styled widget (not system colors)
                    if current_bg in [JackStyle.BG_DARK, JackStyle.BG_MEDIUM, JackStyle.BG_LIGHT, 
                                     DubStyle.BG_DARK, DubStyle.BG_MEDIUM, DubStyle.BG_LIGHT]:
                        # Map old colors to new
                        if current_bg == JackStyle.BG_DARK or current_bg == DubStyle.BG_DARK:
                            widget.configure(bg=self.current_style.BG_DARK)
                        elif current_bg == JackStyle.BG_MEDIUM or current_bg == DubStyle.BG_MEDIUM:
                            widget.configure(bg=self.current_style.BG_MEDIUM)
                        elif current_bg == JackStyle.BG_LIGHT or current_bg == DubStyle.BG_LIGHT:
                            widget.configure(bg=self.current_style.BG_LIGHT)
                        
                        # For Dub style, always use black text
                        if self.current_style.NAME == "Dub":
                            widget.configure(fg="#000000")  # Black text
                        else:
                            # For Jack style, use original text colors
                            if current_fg in [JackStyle.TEXT_PRIMARY, JackStyle.TEXT_SECONDARY,
                                             DubStyle.TEXT_PRIMARY, DubStyle.TEXT_SECONDARY]:
                                if current_fg == JackStyle.TEXT_PRIMARY or current_fg == DubStyle.TEXT_PRIMARY:
                                    widget.configure(fg=self.current_style.TEXT_PRIMARY)
                                elif current_fg == JackStyle.TEXT_SECONDARY or current_fg == DubStyle.TEXT_SECONDARY:
                                    widget.configure(fg=self.current_style.TEXT_SECONDARY)
            
            # Update buttons - force update ALL buttons with full configuration
            elif widget_type == 'Button':
                try:
                    # Skip style toggle button (handled separately)
                    try:
                        widget_text = widget.cget('text')
                        if widget_text in ['Jack', 'Dub']:
                            return  # Skip style toggle
                    except:
                        pass
                    
                    # Force update ALL buttons with full configuration (matching stop button)
                    # For Dub style: gold background with green text
                    if self.current_style.NAME == "Dub":
                        # Apply full configuration multiple times to ensure it sticks
                        for _ in range(3):
                            widget.configure(
                                bg="#FFCC00",
                                fg="#0d1f0d",
                                activebackground="#FFD700",
                                activeforeground="#0d1f0d",
                                highlightbackground="#FFCC00",
                                highlightcolor="#FFCC00",
                                relief=tk.FLAT,
                                borderwidth=0
                            )
                            widget.config(
                                bg="#FFCC00",
                                fg="#0d1f0d",
                                activebackground="#FFD700",
                                activeforeground="#0d1f0d",
                                highlightbackground="#FFCC00",
                                highlightcolor="#FFCC00",
                                relief=tk.FLAT,
                                borderwidth=0
                            )
                    else:
                        # Jack style: light grey background with black text
                        for _ in range(3):
                            widget.configure(
                                bg="#4a4a4a",
                                fg="#000000",
                                activebackground="#525252",
                                activeforeground="#000000",
                                highlightbackground="#4a4a4a",
                                highlightcolor="#4a4a4a",
                                relief=tk.FLAT,
                                borderwidth=0
                            )
                            widget.config(
                                bg="#4a4a4a",
                                fg="#000000",
                                activebackground="#525252",
                                activeforeground="#000000",
                                highlightbackground="#4a4a4a",
                                highlightcolor="#4a4a4a",
                                relief=tk.FLAT,
                                borderwidth=0
                            )
                except Exception:
                    # Skip buttons that can't be configured
                    pass
            
            # Update listbox
            elif widget_type == 'Listbox':
                if hasattr(widget, 'cget'):
                    widget.configure(bg=self.current_style.BG_MEDIUM, fg=self.current_style.TEXT_PRIMARY,
                                   selectbackground=self.current_style.BG_LIGHTER,
                                   selectforeground=self.current_style.TEXT_PRIMARY)
            
            # Update text widgets
            elif widget_type == 'Text':
                if hasattr(widget, 'cget'):
                    widget.configure(bg=self.current_style.BG_MEDIUM, fg=self.current_style.TEXT_PRIMARY)
            
            # Update entry widgets
            elif widget_type == 'Entry':
                if hasattr(widget, 'cget'):
                    widget.configure(bg=self.current_style.BG_MEDIUM, fg=self.current_style.TEXT_PRIMARY,
                                   insertbackground=self.current_style.TEXT_PRIMARY)
            
            # Update scrollbars
            elif widget_type == 'Scrollbar':
                if hasattr(widget, 'cget'):
                    widget.configure(bg=self.current_style.BG_DARK, 
                                   troughcolor=self.current_style.BG_MEDIUM)
            
            # Recursively update children
            for child in widget.winfo_children():
                self._update_widget_style(child)
                
        except Exception:
            # Skip widgets that can't be updated
            pass
    
    def _update_all_buttons(self):
        """Explicitly update all buttons in the GUI - force update regardless of current color."""
        def update_button_recursive(widget):
            """Recursively find and update all buttons."""
            try:
                widget_class = widget.winfo_class()
                if widget_class == 'Button':
                    # Force update ALL buttons regardless of current color
                    # Skip the style toggle button (it's handled separately)
                    try:
                        widget_text = widget.cget('text')
                        if widget_text in ['Jack', 'Dub']:
                            # This is the style toggle, skip it (handled in _toggle_style)
                            pass
                        else:
                            # For Dub style: gold background with green text - match stop button exactly
                            if self.current_style.NAME == "Dub":
                                try:
                                    # Use exact same configuration as _create_button (matching stop button)
                                    widget.configure(
                                        bg="#FFCC00",
                                        fg="#0d1f0d",
                                        activebackground="#FFD700",
                                        activeforeground="#0d1f0d",
                                        highlightbackground="#FFCC00",
                                        highlightcolor="#FFCC00",
                                        relief=tk.FLAT,
                                        borderwidth=0,
                                        overrelief=tk.FLAT,
                                        highlightthickness=0
                                    )
                                    # Also use config() to ensure it sticks
                                    widget.config(
                                        bg="#FFCC00",
                                        fg="#0d1f0d",
                                        activebackground="#FFD700",
                                        activeforeground="#0d1f0d",
                                        highlightbackground="#FFCC00",
                                        highlightcolor="#FFCC00",
                                        relief=tk.FLAT,
                                        borderwidth=0,
                                        overrelief=tk.FLAT,
                                        highlightthickness=0
                                    )
                                    # Force widget update
                                    try:
                                        widget.update_idletasks()
                                    except:
                                        pass
                                except:
                                    # Fallback if some options don't work
                                    try:
                                        widget.config(bg="#FFCC00", fg="#0d1f0d")
                                        widget.configure(bg="#FFCC00", fg="#0d1f0d")
                                    except:
                                        pass
                            else:
                                # Jack style: light grey background with black text
                                try:
                                    # Use exact same configuration as _create_button
                                    widget.configure(
                                        bg="#4a4a4a",
                                        fg="#000000",
                                        activebackground="#525252",
                                        activeforeground="#000000",
                                        highlightbackground="#4a4a4a",
                                        highlightcolor="#4a4a4a",
                                        relief=tk.FLAT,
                                        borderwidth=0
                                    )
                                    # Also use config() to ensure it sticks
                                    widget.config(
                                        bg="#4a4a4a",
                                        fg="#000000",
                                        activebackground="#525252",
                                        activeforeground="#000000",
                                        highlightbackground="#4a4a4a",
                                        highlightcolor="#4a4a4a",
                                        relief=tk.FLAT,
                                        borderwidth=0
                                    )
                                except:
                                    # Fallback if some options don't work
                                    try:
                                        widget.config(bg="#4a4a4a", fg="#000000")
                                        widget.configure(bg="#4a4a4a", fg="#000000")
                                    except:
                                        pass
                    except (tk.TclError, AttributeError):
                        # Button might be destroyed or not support cget
                        # Try to update anyway
                        try:
                            if self.current_style.NAME == "Dub":
                                widget.configure(bg="#FFCC00", fg="#0d1f0d")
                            else:
                                widget.configure(bg="#4a4a4a", fg="#000000")
                        except:
                            pass
            except:
                pass
            
            # Recursively update children
            try:
                for child in widget.winfo_children():
                    update_button_recursive(child)
            except Exception:
                pass
        
        # Start from root and update all buttons - do this multiple times to catch all
        for _ in range(5):  # Try many times to ensure all buttons are caught (macOS needs this)
            update_button_recursive(self.root)
        
        # Force update after recursive update
        try:
            self.root.update_idletasks()
        except:
            pass
        
        # Also explicitly update waveform panel buttons if they exist
        if hasattr(self, 'waveform_panel') and self.waveform_panel:
            try:
                # Update zoom and reset buttons in waveform panel
                if hasattr(self.waveform_panel, '_zoom_in_btn'):
                    if self.current_style.NAME == "Dub":
                        self.waveform_panel._zoom_in_btn.configure(
                            bg="#FFCC00", activebackground="#FFD700", 
                            fg="#0d1f0d", activeforeground="#0d1f0d",
                            highlightbackground="#FFCC00", highlightcolor="#FFCC00",
                            relief=tk.FLAT, borderwidth=0
                        )
                    else:
                        self.waveform_panel._zoom_in_btn.configure(
                            bg="#4a4a4a", activebackground="#525252",
                            fg="#000000", activeforeground="#000000",
                            highlightbackground="#4a4a4a", highlightcolor="#4a4a4a",
                            relief=tk.FLAT, borderwidth=0
                        )
                if hasattr(self.waveform_panel, '_zoom_out_btn'):
                    if self.current_style.NAME == "Dub":
                        self.waveform_panel._zoom_out_btn.configure(
                            bg="#FFCC00", activebackground="#FFD700", 
                            fg="#0d1f0d", activeforeground="#0d1f0d",
                            highlightbackground="#FFCC00", highlightcolor="#FFCC00",
                            relief=tk.FLAT, borderwidth=0
                        )
                    else:
                        self.waveform_panel._zoom_out_btn.configure(
                            bg="#4a4a4a", activebackground="#525252",
                            fg="#000000", activeforeground="#000000",
                            highlightbackground="#4a4a4a", highlightcolor="#4a4a4a",
                            relief=tk.FLAT, borderwidth=0
                        )
                if hasattr(self.waveform_panel, '_reset_btn'):
                    if self.current_style.NAME == "Dub":
                        self.waveform_panel._reset_btn.configure(
                            bg="#FFCC00", activebackground="#FFD700", 
                            fg="#0d1f0d", activeforeground="#0d1f0d",
                            highlightbackground="#FFCC00", highlightcolor="#FFCC00",
                            relief=tk.FLAT, borderwidth=0
                        )
                    else:
                        self.waveform_panel._reset_btn.configure(
                            bg="#4a4a4a", activebackground="#525252",
                            fg="#000000", activeforeground="#000000",
                            highlightbackground="#4a4a4a", highlightcolor="#4a4a4a",
                            relief=tk.FLAT, borderwidth=0
                        )
            except Exception:
                pass
    
    def _apply_styling(self):
        """Apply styling to all widgets."""
        # Configure ttk style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'TProgressbar',
            background=self.current_style.ACCENT_RED,
            troughcolor=self.current_style.BG_MEDIUM,
            borderwidth=0,
            lightcolor=self.current_style.ACCENT_RED,
            darkcolor=self.current_style.ACCENT_RED
        )
        
    def _create_button(self, parent, text, command, color, font=None, state=tk.NORMAL):
        """Create a styled button with appropriate colors for current style."""
        # For Dub style: gold background with green text
        if self.current_style.NAME == "Dub":
            bg_color = "#FFCC00"  # Gold for Dub style
            hover_bg = "#FFD700"  # Brighter gold for hover
            text_color = "#0d1f0d"  # Dark green text for Dub style
        else:
            # Jack style: light grey background with black text
            bg_color = "#4a4a4a"  # Light grey (BG_LIGHTER)
            hover_bg = "#525252"  # Hover state (BG_HOVER)
            text_color = "#000000"  # Black text for Jack style
        
        # On macOS, we need to force custom button rendering by using overrelief
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=text_color,
            font=font or self.current_style.FONT_BODY,
            relief=tk.FLAT,
            bd=0,
            activebackground=hover_bg,
            activeforeground=text_color,
            highlightthickness=0,
            highlightbackground=bg_color,
            highlightcolor=bg_color,
            overrelief=tk.FLAT,  # Prevent native macOS button styling
            state=state,
            cursor="hand2",
            padx=15,
            pady=8
        )
        # Force button to use custom colors (important on macOS) - multiple times to ensure it sticks
        try:
            # Apply all styling options multiple times
            btn.config(
                bg=bg_color,
                fg=text_color,
                activebackground=hover_bg,
                activeforeground=text_color,
                highlightbackground=bg_color,
                highlightcolor=bg_color,
                relief=tk.FLAT,
                borderwidth=0
            )
            btn.configure(
                bg=bg_color,
                fg=text_color,
                activebackground=hover_bg,
                activeforeground=text_color,
                highlightbackground=bg_color,
                highlightcolor=bg_color,
                relief=tk.FLAT,
                borderwidth=0
            )
            # Force one more time
            btn.config(bg=bg_color, fg=text_color)
        except:
            pass
        
        # Store button reference for explicit updates
        self.all_buttons.append(btn)
        
        # Immediately apply full styling if in Dub mode (ensures buttons are gold from start)
        if self.current_style.NAME == "Dub":
            self._reapply_button_style(btn)
        
        return btn
        
    def _load_files(self):
        """Load audio files."""
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.aif *.aiff"),
                ("WAV Files", "*.wav"),
                ("MP3 Files", "*.mp3"),
                ("FLAC Files", "*.flac"),
                ("AIFF Files", "*.aif *.aiff"),
                ("All Files", "*.*")
            ]
        )
        
        if files:
            new_files = []
            for file_path in files:
                path = Path(file_path)
                # Note: Large file checking is now done during analysis with user prompt
                # The skip_large option is kept for backward compatibility but large files
                # will be handled with a warning dialog during analysis
                if path not in self.file_paths:
                    self.file_paths.append(path)
                    new_files.append(path.name)
            
            # Update listbox
            self._update_file_list()
            
            # Re-apply button styling after file list update (this is why buttons become gold when loading)
            self._force_update_all_buttons()
            
            if new_files:
                self._log_result(f"LOADED: {len(new_files)} file(s)\n")
                
    def _remove_selected(self):
        """Remove selected file from list."""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            removed = self.file_paths.pop(index)
            # Remove from audio samples cache
            self.audio_samples.pop(removed, None)
            self._update_file_list()
            if self.waveform_panel:
                self.waveform_panel.clear()
            self._log_result(f"REMOVED: {removed.name}\n")
            
    def _select_all(self):
        """Select all files in the list."""
        self.file_listbox.selection_set(0, tk.END)
        count = len(self.file_paths)
        self._log_result(f"Selected all {count} file(s)\n")
        
    def _deselect_all(self):
        """Deselect all files in the list."""
        self.file_listbox.selection_clear(0, tk.END)
        self._log_result("Deselected all files\n")
        
    def _clear_all(self):
        """Clear all files from list."""
        if self.file_paths:
            count = len(self.file_paths)
            self.file_paths.clear()
            self.audio_samples.clear()
            self._update_file_list()
            if self.waveform_panel:
                self.waveform_panel.clear()
            self._log_result(f"CLEARED: {count} file(s)\n")
            
    def _update_file_list(self):
        """Update file listbox with current file paths."""
        items = [f"{i+1}. {path.name}" for i, path in enumerate(self.file_paths)]
        self.file_listbox.set_items(items)

    def _update_analyze_button(self):
        """Update analyze button text based on selection."""
        if not hasattr(self, 'analyze_button'):
            return
        selection = self.file_listbox.curselection()
        if selection:
            count = len(selection)
            self.analyze_button.config(text=f"Analyze Selected ({count})")
        else:
            self.analyze_button.config(text="Analyze All")
        # Re-apply styling after config change to ensure colors persist
        self._reapply_button_style(self.analyze_button)
    
    def _on_file_select(self, event):
        """Handle file selection to display waveform."""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        # Re-apply button styling after selection (prevents buttons from losing gold)
        self._force_update_all_buttons()

        index = selection[0]
        # Extract original index from listbox item (handles reordering)
        try:
            item = self.file_listbox.get(index)
            original_index = int(item.split('.')[0]) - 1
            if 0 <= original_index < len(self.file_paths):
                file_path = self.file_paths[original_index]
                self._load_waveform_for_file(file_path)
        except (ValueError, IndexError):
            pass

    def _load_waveform_for_file(self, file_path: Path):
        """Load and display waveform for a file."""
        if self.waveform_panel is None or self.loader is None:
            return

        # Check if we have a cached audio sample
        if file_path in self.audio_samples:
            audio_sample = self.audio_samples[file_path]
            self.waveform_panel.set_from_audio_sample(audio_sample)
            return

        # Load audio in background thread to avoid UI freeze
        def load_audio():
            try:
                audio_sample = self.loader.load(file_path)
                # Cache the sample
                self.audio_samples[file_path] = audio_sample
                # Update UI on main thread
                self.root.after(0, lambda: self._display_waveform(audio_sample))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self._log_result(f"Waveform load error: {msg}\n"))

        thread = threading.Thread(target=load_audio, daemon=True)
        thread.start()

    def _display_waveform(self, audio_sample):
        """Display waveform from audio sample (called on main thread)."""
        if self.waveform_panel is not None:
            self.waveform_panel.set_from_audio_sample(audio_sample)

    def _start_analysis(self):
        """Start analysis process."""
        if not self.file_paths:
            messagebox.showwarning("No Files", "Please load files first.")
            return
            
        if self.processing:
            return
            
        # Get reordered files from listbox
        items = self.file_listbox.get_items()
        reordered_paths = []
        for item in items:
            # Extract index from "1. filename" format
            try:
                index = int(item.split('.')[0]) - 1
                if 0 <= index < len(self.file_paths):
                    reordered_paths.append(self.file_paths[index])
            except (ValueError, IndexError):
                continue
                
        # Use original order if reordering failed
        if not reordered_paths:
            reordered_paths = self.file_paths.copy()
            
        # Get selected files (if any)
        selected_indices = self.file_listbox.curselection()
        if selected_indices:
            # Only analyze selected files
            selected_paths = [reordered_paths[i] for i in selected_indices if 0 <= i < len(reordered_paths)]
            if not selected_paths:
                messagebox.showwarning("No Selection", "Please select files to analyze.")
                return
            files_to_analyze = selected_paths
            self._log_result(f"Analyzing {len(files_to_analyze)} selected file(s)...\n")
        else:
            # Analyze all files
            files_to_analyze = reordered_paths
            self._log_result(f"Analyzing all {len(files_to_analyze)} file(s)...\n")
        
        # Log which files will be analyzed
        if len(files_to_analyze) > 0:
            file_names = [f.name for f in files_to_analyze]
            self._log_result(f"Files to analyze: {', '.join(file_names)}\n")
        
        # Check for large files (> 10MB) and prompt user
        large_files = []
        for path in files_to_analyze:
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > 10:
                    large_files.append((path, size_mb))
                    self._log_result(f"LARGE FILE DETECTED: {path.name} ({size_mb:.1f} MB)\n")
                    # Also print to terminal for debugging
                    print(f"LARGE FILE DETECTED: {path.name} ({size_mb:.1f} MB)")
            except Exception as e:
                self._log_result(f"Error checking file size for {path.name}: {e}\n")
        
        # If there are large files, ask user what to do
        if large_files:
            self._log_result(f"Found {len(large_files)} large file(s), showing warning dialog...\n")
            print(f"Found {len(large_files)} large file(s), showing warning dialog...")
            try:
                if len(large_files) == 1:
                    # Single large file
                    path, size_mb = large_files[0]
                    self._log_result(f"Prompting user about large file: {path.name} ({size_mb:.1f} MB)\n")
                    # Bring window to front before showing dialog
                    self.root.lift()
                    self.root.focus_force()
                    response = messagebox.askyesno(
                        "Large File Detected",
                        f"File '{path.name}' is {size_mb:.1f} MB (>{10} MB).\n\n"
                        f"Large files may take longer to analyze.\n\n"
                        f"Proceed with analysis?",
                        icon='warning',
                        parent=self.root
                    )
                    self._log_result(f"User response: {'Yes' if response else 'No'}\n")
                    print(f"User response to large file dialog: {'Yes' if response else 'No'}")
                    if not response:
                        # User chose not to proceed, remove from list
                        files_to_analyze = [f for f in files_to_analyze if f != path]
                        self._log_result(f"SKIPPED: {path.name} (user declined large file analysis)\n")
                        print(f"SKIPPED: {path.name} (user declined large file analysis)")
                    else:
                        self._log_result(f"PROCEEDING: {path.name} will be analyzed\n")
                        print(f"PROCEEDING: {path.name} will be analyzed")
                else:
                    # Multiple large files
                    large_file_names = [f"{path.name} ({size_mb:.1f} MB)" for path, size_mb in large_files]
                    large_files_list = "\n".join(large_file_names[:5])  # Show first 5
                    if len(large_file_names) > 5:
                        large_files_list += f"\n... and {len(large_file_names) - 5} more"
                    
                    self._log_result(f"Prompting user about {len(large_files)} large files\n")
                    # Bring window to front before showing dialog
                    self.root.lift()
                    self.root.focus_force()
                    response = messagebox.askyesno(
                        "Large Files Detected",
                        f"{len(large_files)} file(s) are > 10 MB:\n\n"
                        f"{large_files_list}\n\n"
                        f"Large files may take longer to analyze.\n\n"
                        f"Proceed with analysis of all {len(large_files)} large file(s)?",
                        icon='warning',
                        parent=self.root
                    )
                    self._log_result(f"User response: {'Yes' if response else 'No'}\n")
                    print(f"User response to large files dialog: {'Yes' if response else 'No'}")
                    if not response:
                        # User chose not to proceed, remove all large files from list
                        large_file_paths = [path for path, _ in large_files]
                        files_to_analyze = [f for f in files_to_analyze if f not in large_file_paths]
                        skipped_names = [path.name for path in large_file_paths]
                        self._log_result(f"SKIPPED: {len(skipped_names)} large file(s) (user declined)\n")
                        print(f"SKIPPED: {len(skipped_names)} large file(s) (user declined)")
                    else:
                        self._log_result(f"PROCEEDING: All {len(large_files)} large file(s) will be analyzed\n")
                        print(f"PROCEEDING: All {len(large_files)} large file(s) will be analyzed")
            except Exception as e:
                # If dialog fails, log error but proceed with analysis anyway
                error_msg = str(e)
                self._log_result(f"ERROR showing large file dialog: {error_msg}\n")
                self._log_result(f"PROCEEDING with analysis anyway (assuming user wants to continue)\n")
                print(f"ERROR showing large file dialog: {error_msg}")
                print("PROCEEDING with analysis anyway (assuming user wants to continue)")
                import traceback
                traceback.print_exc()
        
        # Check if we still have files to analyze after filtering
        if not files_to_analyze:
            self._log_result("ERROR: No files remaining after filtering large files\n")
            messagebox.showinfo("No Files", "No files selected for analysis after filtering large files.")
            return
        
        # Log final file list
        self._log_result(f"Final files to analyze: {len(files_to_analyze)} file(s)\n")
        print(f"Final files to analyze: {len(files_to_analyze)} file(s)")
        for f in files_to_analyze:
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
                self._log_result(f"  - {f.name} ({size_mb:.1f} MB)\n")
                print(f"  - {f.name} ({size_mb:.1f} MB)")
            except:
                self._log_result(f"  - {f.name}\n")
                print(f"  - {f.name}")
        
        # Update file_paths with reordered list
        self.file_paths = reordered_paths
        
        # Disable controls
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        # Re-apply styling after config change to ensure colors persist
        self._reapply_button_style(self.analyze_button)
        self._reapply_button_style(self.stop_button)
        self.processing = True
        
        # Log that we're starting analysis
        self._log_result(f"Starting analysis thread with {len(files_to_analyze)} file(s)...\n")
        print(f"Starting analysis thread with {len(files_to_analyze)} file(s)...")
        
        # Start analysis in thread
        thread = threading.Thread(target=self._run_analysis, args=(files_to_analyze,), daemon=True)
        thread.start()
        print(f"Analysis thread started (thread ID: {thread.ident})")
        
    def _run_analysis(self, file_paths: List[Path]):
        """Run analysis on files."""
        print(f"_run_analysis called with {len(file_paths)} file(s)")
        total = len(file_paths)
        mode = self.processing_mode.get()
        llm_enabled = self.llm_enabled.get()
        speech_enabled = self.speech_enabled.get()
        print(f"Processing mode: {mode}, LLM enabled: {llm_enabled}, Speech enabled: {speech_enabled}")

        # Temporarily disable LLM if needed
        original_llm = None
        if not llm_enabled and 'llm' in self.engine.analyzers:
            original_llm = self.engine.analyzers.pop('llm', None)

        # Temporarily disable Speech if needed (Whisper can be slow on large files)
        original_speech = None
        if not speech_enabled and 'speech' in self.engine.analyzers:
            original_speech = self.engine.analyzers.pop('speech', None)
        
        try:
            # Process files one by one (both batch and sequential modes now display results immediately)
            files_processed = 0
            for i, path in enumerate(file_paths):
                if not self.processing:  # Check if stopped
                    break
                    
                self._update_progress(i / total * 100, f"Processing {i + 1}/{total}: {path.name}")
                
                try:
                    # Cache audio sample for waveform display if not already cached
                    if path not in self.audio_samples and self.loader:
                        try:
                            audio_sample = self.loader.load(path)
                            self.audio_samples[path] = audio_sample
                        except Exception:
                            pass  # Non-critical, continue with analysis

                    # Analyze file
                    print(f"Calling engine.analyze() for {path.name}...")
                    result = self.engine.analyze(path)
                    print(f"Analysis complete for {path.name}, result: {'present' if result else 'None'}")
                    files_processed += 1
                    
                    # Store and display result immediately
                    if result:
                        self.results[path] = result
                        # Add to table instead of text display
                        audio_sample = self.audio_samples.get(path)
                        # Use default parameter to capture values correctly
                        def add_result(p=path, r=result, a=audio_sample):
                            self._add_to_table(p, r, a)
                        self.root.after(0, add_result)
                        
                        # Auto-display waveform for first analyzed file
                        if i == 0 and path in self.audio_samples:
                            self.root.after(0, lambda p=path: self._display_waveform(self.audio_samples.get(p)))
                    else:
                        self._log_result(f"WARNING: {path.name} - Analysis returned no result\n")
                        
                except Exception as e:
                    error_msg = str(e)
                    file_name = path.name
                    self.root.after(0, lambda msg=error_msg, name=file_name: self._log_result(f"ERROR: {name} - {msg}\n"))
                    continue  # Continue to next file even if this one failed
                    
                # Update progress after each file
                self._update_progress((i + 1) / total * 100, f"Completed {i + 1}/{total}")
            
            # Analysis loop completed successfully
            print(f"Analysis loop completed. Processed {files_processed}/{total} file(s).")
            self._log_result(f"Analysis completed: processed {files_processed}/{total} file(s)\n")
            # Mark processing as complete before finally block
            self.processing = False
            print("Set self.processing = False (analysis actually complete)")
                    
        except Exception as e:
            error_msg = str(e)
            print(f"ANALYSIS ERROR in _run_analysis: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda msg=error_msg: self._log_result(f"ANALYSIS ERROR: {msg}\n"))
        finally:
            # Restore LLM analyzer
            if original_llm:
                self.engine.analyzers['llm'] = original_llm
                print("LLM analyzer restored")

            # Restore Speech analyzer
            if original_speech:
                self.engine.analyzers['speech'] = original_speech
                print("Speech analyzer restored")
            
            # Only call _analysis_complete after the entire analysis is done
            # This will be called when the try block completes (successfully or with exception)
            print("Finally block executing - calling _analysis_complete()")
            # Use after_idle to ensure this happens after all other pending operations
            self.root.after_idle(self._analysis_complete)
            
    def _analysis_complete(self):
        """Called when analysis completes."""
        print("_analysis_complete() called - checking if analysis is actually done")
        
        # Double-check that we're not still processing
        # This can happen if _analysis_complete is called prematurely
        if self.processing:
            print("WARNING: _analysis_complete() called but self.processing is still True")
            print("This suggests analysis is still running - not updating UI yet")
            # Don't update UI if analysis is still running
            # Schedule a check again in a moment
            self.root.after(1000, self._check_analysis_status)
            return
        
        print("_analysis_complete() - analysis is done, updating UI")
        self.processing = False
        self.analyze_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # Re-apply styling after config change to ensure colors persist
        self._reapply_button_style(self.analyze_button)
        self._reapply_button_style(self.stop_button)
        
        # Only show "Analysis complete!" if we actually finished processing
        # Check if we have results to determine if analysis actually completed
        if self.results:
            completed_count = len(self.results)
            self._update_progress(100, f"Analysis complete! ({completed_count} file(s) analyzed)")
            print(f"Analysis complete! ({completed_count} file(s) analyzed)")
        else:
            # Analysis was stopped or failed
            self._update_progress(100, "Analysis stopped")
            print("Analysis stopped (no results)")
    
    def _check_analysis_status(self):
        """Check if analysis is still running and update UI accordingly."""
        if not self.processing:
            # Analysis actually completed, update UI now
            self._analysis_complete()
        else:
            # Still processing, update progress but don't mark as complete
            # Keep the current progress message
            pass
        
    def _stop_analysis(self):
        """Stop analysis process."""
        self.processing = False
        self._log_result("ANALYSIS STOPPED BY USER\n")
        self._analysis_complete()
        
    def _update_progress(self, value: float, message: str):
        """Update progress bar and label."""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=message))
        
    def _display_result(self, path: Path, result: AnalysisResult):
        """Display analysis result."""
        output = f"\n{'='*60}\n"
        output += f"FILE: {path.name}\n"
        output += f"{'='*60}\n"
        output += f"Processing Time: {result.processing_time:.3f}s\n"
        output += f"Quality: {result.quality_metadata.get('quality_tier', 'Unknown')}\n"
        output += f"\n{result.get_summary()}\n"
        
        if result.source_classification:
            output += f"\nSource: {result.source_classification.source_type} "
            output += f"({result.source_classification.confidence:.1%})\n"
            
        if result.musical_analysis and result.musical_analysis.has_pitch:
            output += f"Musical: {result.musical_analysis.note_name} "
            output += f"({result.musical_analysis.fundamental_frequency:.1f} Hz)\n"
            
        if result.percussive_analysis:
            output += f"Percussive: {result.percussive_analysis.drum_type} "
            output += f"({result.percussive_analysis.confidence:.1%})\n"
            
        if result.rhythmic_analysis:
            if result.rhythmic_analysis.has_rhythm:
                output += f"Rhythm: {result.rhythmic_analysis.tempo_bpm:.1f} BPM\n"
            output += f"One-Shot: {result.rhythmic_analysis.is_one_shot}\n"
            
        if result.llm_analysis:
            output += f"\n{'='*60}\n"
            output += f"LLM ANALYSIS:\n"
            output += f"{'='*60}\n"
            output += f"  Name: {result.llm_analysis.suggested_name}\n"
            output += f"  Description: {result.llm_analysis.description}\n"
            
            # Prominently display speech transcription if available
            if result.llm_analysis.contains_speech or result.llm_analysis.transcription:
                output += f"\n  --- SPEECH RECOGNITION ---\n"
                output += f"  Contains Speech: Yes\n"
                if result.llm_analysis.transcription:
                    # Display full transcription prominently
                    transcription = result.llm_analysis.transcription
                    # Wrap long transcriptions for readability
                    if len(transcription) > 80:
                        words = transcription.split()
                        lines = []
                        current_line = []
                        current_length = 0
                        for word in words:
                            if current_length + len(word) + 1 > 80 and current_line:
                                lines.append(" ".join(current_line))
                                current_line = [word]
                                current_length = len(word)
                            else:
                                current_line.append(word)
                                current_length += len(word) + 1
                        if current_line:
                            lines.append(" ".join(current_line))
                        output += f"  Transcription:\n"
                        for line in lines:
                            output += f"    {line}\n"
                    else:
                        output += f"  Transcription: {transcription}\n"
                if result.llm_analysis.detected_words:
                    output += f"  Detected Words: {', '.join(result.llm_analysis.detected_words[:20])}"
                    if len(result.llm_analysis.detected_words) > 20:
                        output += f" ... ({len(result.llm_analysis.detected_words) - 20} more)"
                    output += "\n"
                if result.llm_analysis.speech_language:
                    output += f"  Language: {result.llm_analysis.speech_language}\n"
                if result.llm_analysis.speech_confidence is not None:
                    output += f"  Speech Confidence: {result.llm_analysis.speech_confidence:.2%}\n"
            else:
                output += f"  Contains Speech: No\n"
            
            if result.llm_analysis.tags:
                output += f"\n  Tags: {', '.join(result.llm_analysis.tags)}\n"
            output += f"  Model: {result.llm_analysis.model_used}\n"
            
        output += "\n"
        self._log_result(output)
        
        # Prominently display full transcription and description
        if result.llm_analysis:
            if result.llm_analysis.transcription:
                self._log_result(f"\n{'='*70}\n")
                self._log_result(f"FULL TRANSCRIPTION - {path.name}\n")
                self._log_result(f"{'='*70}\n")
                self._log_result(f"{result.llm_analysis.transcription}\n")
                self._log_result(f"{'='*70}\n\n")
            if result.llm_analysis.description:
                self._log_result(f"{'='*70}\n")
                self._log_result(f"FULL DESCRIPTION - {path.name}\n")
                self._log_result(f"{'='*70}\n")
                self._log_result(f"{result.llm_analysis.description}\n")
                self._log_result(f"{'='*70}\n\n")

        # Update run button visibility now that results exist
        self._update_run_button_visibility()

    def _add_to_table(self, path: Path, result: AnalysisResult, audio_sample=None):
        """
        Add result to table display.
        
        Args:
            path: Path to audio file
            result: Analysis result
            audio_sample: Optional AudioSample for thumbnail
        """
        if self.results_table:
            self.results_table.add_result(path, result, audio_sample)
    
    def _on_table_row_select(self, path: Path):
        """Handle table row selection to display waveform."""
        if path in self.audio_samples:
            self._display_waveform(self.audio_samples[path])
        # Re-apply button styling after selection (prevents buttons from losing gold)
        self._force_update_all_buttons()
        
    def _log_result(self, text: str):
        """Add text to raw output display."""
        if hasattr(self, 'output_text') and self.output_text:
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)  # Auto-scroll to bottom

    def _toggle_ai_section(self):
        """Toggle visibility of the AI Features section."""
        if self.ai_section_visible.get():
            self.ai_features_frame.pack_forget()
            self.ai_section_visible.set(False)
        else:
            self.ai_features_frame.pack(fill=tk.X, after=self.ai_toggle_label.master)
            self.ai_section_visible.set(True)
        self._update_ai_header()

    def _sync_feature_toggle(self, feature_id: str):
        """Sync a checkbox state to the FeatureGate and update UI."""
        if self.feature_manager is not None:
            enabled = self.ai_feature_vars[feature_id].get()
            self.feature_manager._gate.set_enabled(feature_id, enabled)
        # Dim/undim the checkbox label
        if feature_id in self.ai_feature_checks:
            cb = self.ai_feature_checks[feature_id]
            if self.ai_feature_vars[feature_id].get():
                cb.config(fg=self.current_style.TEXT_PRIMARY)
            else:
                cb.config(fg=self.current_style.TEXT_DISABLED)
        self._update_ai_header()
        self._update_run_button_visibility()

    def _update_ai_header(self):
        """Update the AI Features header with enabled count."""
        total = len(self.ai_feature_vars)
        enabled = sum(1 for v in self.ai_feature_vars.values() if v.get())
        arrow = "\u25bc" if self.ai_section_visible.get() else "\u25b6"
        self.ai_toggle_label.config(text=f"{arrow} AI Features ({enabled}/{total})")

    def _update_run_button_visibility(self):
        """Show/hide the Run AI Features button based on state."""
        enabled = sum(1 for v in self.ai_feature_vars.values() if v.get())
        has_results = bool(self.results)
        if enabled > 0 and has_results:
            self.ai_run_btn_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        else:
            self.ai_run_btn_frame.pack_forget()

    def _build_feature_kwargs(self, feature_id: str) -> dict:
        """Build kwargs for a feature from the GUI parameter inputs."""
        kwargs: dict = {}
        if feature_id == "natural_language_search":
            query = self.ai_nl_query.get().strip()
            if query:
                kwargs["query"] = query
        elif feature_id == "daw_suggestions":
            from src.features.models import DAWContext
            try:
                bpm = float(self.ai_daw_bpm.get().strip() or "120")
            except ValueError:
                bpm = 120.0
            kwargs["context"] = DAWContext(
                bpm=bpm,
                key=self.ai_daw_key.get().strip() or "C major",
                genre=self.ai_daw_genre.get().strip() or "electronic",
                mood="neutral",
            )
        elif feature_id == "marketplace_description":
            kwargs["pack_name"] = self.ai_pack_name.get().strip() or "Untitled Pack"
        elif feature_id == "sample_chain":
            kwargs["energy_preference"] = self.ai_energy_pref.get()
        elif feature_id == "batch_rename":
            kwargs["dry_run"] = True
        return kwargs

    @staticmethod
    def _summarize_feature_data(data) -> tuple:
        """
        Produce (summary, detail) strings from a feature result dataclass.

        summary  → short one-liner for the Description column
        detail   → longer text for the Transcription column / raw output
        """
        from dataclasses import fields as dc_fields

        if not hasattr(data, '__dataclass_fields__'):
            return (str(data)[:120], str(data))

        parts_short: list = []
        parts_long: list = []
        for f in dc_fields(data):
            val = getattr(data, f.name)
            if val is None or val == "" or val == []:
                continue
            if isinstance(val, list):
                # Compact: first 3 items
                preview = ", ".join(str(v)[:60] for v in val[:3])
                if len(val) > 3:
                    preview += f" (+{len(val)-3})"
                parts_short.append(f"{f.name}: [{preview}]")
                full = "\n  ".join(str(v) for v in val)
                parts_long.append(f"{f.name}:\n  {full}")
            elif isinstance(val, dict):
                parts_short.append(f"{f.name}: {len(val)} items")
                items = "\n  ".join(f"{k}: {v}" for k, v in list(val.items())[:10])
                parts_long.append(f"{f.name}:\n  {items}")
            elif isinstance(val, float):
                parts_short.append(f"{f.name}: {val:.2f}")
                parts_long.append(f"{f.name}: {val:.4f}")
            else:
                s = str(val)
                parts_short.append(f"{f.name}: {s[:80]}")
                parts_long.append(f"{f.name}: {s}")

        summary = " | ".join(parts_short)[:300]
        detail = "\n".join(parts_long)
        return summary, detail

    def _run_ai_features(self):
        """Run all enabled AI features and display results as table rows."""
        if self.feature_manager is None:
            self._log_result("\n[AI Features] No feature manager available. "
                             "Check LLM configuration in config.yaml.\n")
            return
        if not self.results:
            self._log_result("\n[AI Features] No analysis results available. "
                             "Run analysis first.\n")
            return

        enabled_ids = [
            fid for fid, var in self.ai_feature_vars.items() if var.get()
        ]
        if not enabled_ids:
            self._log_result("\n[AI Features] No features enabled.\n")
            return

        # Clear previous AI feature rows
        if self.results_table:
            self.results_table.clear_feature_rows()

        self._log_result(f"\n{'='*60}\n"
                         f"RUNNING AI FEATURES ({len(enabled_ids)} enabled)\n"
                         f"{'='*60}\n")

        all_results = list(self.results.values())

        def _run_in_background():
            for feature_id in enabled_ids:
                display_name = self.ai_feature_names.get(
                    feature_id, feature_id.replace("_", " ").title()
                )
                kwargs = self._build_feature_kwargs(feature_id)
                try:
                    result = self.engine.run_feature(feature_id, all_results, **kwargs)
                    summary, detail = self._summarize_feature_data(result.data)
                    time_str = f"{result.processing_time:.2f}s"
                    model = result.model_used or ""

                    # Insert row in table
                    def _add_row(dn=display_name, s=summary, d=detail,
                                 t=time_str, m=model, fid=feature_id):
                        if self.results_table:
                            self.results_table.add_feature_row(
                                feature_name=dn, summary=s, detail=d,
                                time_str=t, model=m, tag_id=fid,
                            )
                    self.root.after(0, _add_row)

                    # Also append to raw output
                    raw = (f"\n--- {display_name} ({time_str}) ---\n"
                           f"{detail}\n")
                    self.root.after(0, lambda r=raw: self._log_result(r))

                except Exception as e:
                    err = f"\n--- {display_name} ---\n  Error: {e}\n"
                    self.root.after(0, lambda m=err: self._log_result(m))

            self.root.after(0, lambda: self._log_result(
                f"\n{'='*60}\nAI FEATURES COMPLETE\n{'='*60}\n\n"))

        thread = threading.Thread(target=_run_in_background, daemon=True)
        thread.start()
        
    def _clear_results(self):
        """Clear results display."""
        if self.results_table:
            self.results_table.clear()
        self.results.clear()
        if self.waveform_panel:
            self.waveform_panel.clear()
        if hasattr(self, 'output_text') and self.output_text:
            self.output_text.delete('1.0', tk.END)
        if hasattr(self, 'search_var'):
            self.search_var.set("")
    
    def _on_search_change(self, *args):
        """Handle search text change."""
        search_text = self.search_var.get()
        if self.results_table:
            self.results_table.filter_by_text(search_text)
    
    def _on_tag_click(self, tag: str):
        """Handle tag click to filter by tag."""
        if self.results_table:
            self.results_table.filter_by_tag(tag)
            # Update search field to show active tag filter
            self.search_var.set(f"tag:{tag}")
        
    def _export_results(self):
        """Export results to file."""
        if not self.results:
            messagebox.showwarning("No Results", "No results to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # Export as JSON
                    import json
                    results_dict = {}
                    for path, result in self.results.items():
                        # Convert result to dict if it has to_dict method, otherwise use to_json
                        if hasattr(result, 'to_dict'):
                            result_dict = result.to_dict()
                        else:
                            # Parse JSON string if to_dict not available
                            result_dict = json.loads(result.to_json())
                        
                        # Add LLM description and transcription if present
                        if result.llm_analysis:
                            if result.llm_analysis.description:
                                result_dict['llm_description'] = result.llm_analysis.description
                            if result.llm_analysis.transcription:
                                result_dict['transcription'] = result.llm_analysis.transcription
                        
                        results_dict[str(path)] = result_dict
                    with open(file_path, 'w') as f:
                        json.dump(results_dict, f, indent=2)
                else:
                    # Export as text
                    with open(file_path, 'w') as f:
                        f.write("SAMPLECHARM ANALYSIS RESULTS\n")
                        f.write("=" * 60 + "\n\n")
                        for path, result in self.results.items():
                            f.write(f"File: {path}\n")
                            f.write(f"Time: {result.processing_time:.3f}s\n")
                            f.write(f"{result.get_summary()}\n")
                            
                            # Include LLM description if present
                            if result.llm_analysis and result.llm_analysis.description:
                                f.write(f"\nDescription:\n")
                                f.write(f"{result.llm_analysis.description}\n")
                            
                            # Include transcription if present
                            if result.llm_analysis and result.llm_analysis.transcription:
                                f.write(f"\nTranscription:\n")
                                f.write(f"{result.llm_analysis.transcription}\n")
                            
                            f.write("\n" + "-" * 60 + "\n\n")
                            
                messagebox.showinfo("Export Complete", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    app = SampleCharmGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
