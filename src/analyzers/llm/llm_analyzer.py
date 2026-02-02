"""
LLM-based audio analyzer for naming, description, and speech detection.

Supports TogetherAI and OpenAI as LLM providers.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, LLMAnalysis
from src.utils.errors import AnalysisError, ModelLoadError


# Default models for each provider
DEFAULT_MODELS = {
    'togetherai': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'openai': 'gpt-4o-mini'
}


class LLMAnalyzer(BaseAnalyzer[LLMAnalysis]):
    """
    LLM-based analyzer for audio naming, description, and speech detection.

    Uses TogetherAI or OpenAI to analyze audio characteristics and generate:
    - A suggested name for the audio sample
    - A detailed description
    - Speech/word detection
    - Categorization tags
    """

    def __init__(
        self,
        provider: str = 'togetherai',
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
        shared_client=None,
    ):
        """
        Initialize LLM analyzer.

        Args:
            provider: LLM provider ('togetherai' or 'openai')
            model: Model name (uses default if not specified)
            api_key: API key (uses environment variable if not specified)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            shared_client: Optional LLMClient from src.features.client.
                           When provided, API calls delegate to this client
                           instead of creating a separate OpenAI instance.
        """
        super().__init__("llm_analyzer", "1.0.0")

        self._shared_client = shared_client
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS['togetherai'])
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from environment if not provided
        if api_key:
            # If api_key is still a template string (${VAR}), try to resolve it
            if api_key.startswith('${') and api_key.endswith('}'):
                var_name = api_key[2:-1]
                self.api_key = os.environ.get(var_name)
            else:
                self.api_key = api_key
        else:
            self.api_key = None

        # If still no key, try environment variables
        if not self.api_key:
            if self.provider == 'togetherai':
                # Try both common environment variable names
                self.api_key = os.environ.get('TOGETHER_API_KEY') or os.environ.get('TOGETHERAI_API_KEY')
            elif self.provider == 'openai':
                self.api_key = os.environ.get('OPENAI_API_KEY')

        self._client = None

    @property
    def client(self):
        """Lazy-load the API client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create the appropriate API client."""
        if not self.api_key:
            env_vars = 'TOGETHER_API_KEY or TOGETHERAI_API_KEY' if self.provider == 'togetherai' else 'OPENAI_API_KEY'
            raise ModelLoadError(
                f"No API key found for {self.provider}. "
                f"Set {env_vars} environment variable.",
                model_name=self.provider
            )

        if self.provider == 'togetherai':
            try:
                from openai import OpenAI
                return OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.together.xyz/v1"
                )
            except ImportError:
                raise ModelLoadError(
                    "openai package required for TogetherAI. Install with: pip install openai",
                    model_name="togetherai"
                )

        elif self.provider == 'openai':
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                raise ModelLoadError(
                    "openai package required. Install with: pip install openai",
                    model_name="openai"
                )

        else:
            raise ModelLoadError(
                f"Unknown provider: {self.provider}. Use 'togetherai' or 'openai'.",
                model_name=self.provider
            )

    def _analyze_impl(self, audio: AudioSample, speech_data: Optional[dict] = None, rhythmic_data=None) -> LLMAnalysis:
        """
        Analyze audio using LLM.

        Args:
            audio: AudioSample to analyze
            speech_data: Optional speech recognition results from Whisper
            rhythmic_data: Optional RhythmicAnalysis from rhythmic analyzer

        Returns:
            LLMAnalysis: LLM-generated analysis (always returns a result, never None)
        """
        # Build context from audio characteristics
        context = self._build_audio_context(audio)

        # Enhance context with rhythmic data if available
        if rhythmic_data is not None:
            if hasattr(rhythmic_data, 'beat_times') and rhythmic_data.beat_times:
                beat_times_ms = [round(t * 1000, 1) for t in rhythmic_data.beat_times]
                context['beat_timestamps_ms'] = beat_times_ms[:20]
                context['num_beats'] = len(rhythmic_data.beat_times)
            if hasattr(rhythmic_data, 'tempo_bpm') and rhythmic_data.tempo_bpm:
                context['tempo_bpm'] = round(rhythmic_data.tempo_bpm, 1)
            if hasattr(rhythmic_data, 'is_one_shot'):
                context['is_one_shot'] = rhythmic_data.is_one_shot

        # Enhance context with speech data if available
        if speech_data:
            context['speech_detected'] = speech_data.get('contains_speech', False)
            transcription = speech_data.get('transcription', '').strip()
            context['transcription'] = transcription
            context['speech_words'] = speech_data.get('detected_words', [])
            context['speech_language'] = speech_data.get('speech_language', None)
            context['speech_confidence'] = speech_data.get('confidence', 0.0)
            # Log for debugging
            self.logger.debug(f"Speech data added to context: has_transcription={bool(transcription)}, contains_speech={context['speech_detected']}, words_count={len(context['speech_words'])}")

        # Create prompt for LLM
        prompt = self._create_analysis_prompt(context)

        # Call LLM API
        response = self._call_llm(prompt)

        # Parse response into structured result
        result = self._parse_response(response, speech_data, context)

        return result

    def _build_audio_context(self, audio: AudioSample) -> Dict[str, Any]:
        """
        Build context dictionary from audio characteristics.

        Args:
            audio: AudioSample with features

        Returns:
            Dict containing audio characteristics for the prompt
        """
        # Basic info
        context = {
            'file_name': audio.file_path.stem,
            'duration': f"{audio.duration:.2f} seconds",
            'sample_rate': f"{audio.original_sample_rate} Hz",
            'bit_depth': audio.original_bit_depth,
            'channels': 'stereo' if audio.channels == 2 else 'mono',
            'format': audio.original_format,
            'quality_tier': audio.quality_info.get('quality_tier', 'Unknown')
        }

        # Extract feature summaries
        try:
            features = audio.features

            # Spectral characteristics
            spectral_centroid_mean = float(np.mean(features.spectral_centroid))
            context['brightness'] = 'bright' if spectral_centroid_mean > 3000 else 'warm' if spectral_centroid_mean < 1500 else 'balanced'

            # Zero crossing rate (indicates noisiness/percussion)
            zcr_mean = float(np.mean(features.zero_crossing_rate))
            context['texture'] = 'noisy/percussive' if zcr_mean > 0.1 else 'smooth/tonal'

            # Harmonic vs percussive energy
            harmonic_energy = float(np.sum(features.harmonic ** 2))
            percussive_energy = float(np.sum(features.percussive ** 2))
            total_energy = harmonic_energy + percussive_energy
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
                context['character'] = 'harmonic/melodic' if harmonic_ratio > 0.6 else 'percussive/rhythmic' if harmonic_ratio < 0.4 else 'mixed'
            else:
                context['character'] = 'quiet/silent'

            # Onset count (indicates complexity)
            onset_count = len(features.onset_frames)
            context['complexity'] = f"{onset_count} onsets detected"

            # Onset timestamps in milliseconds
            if onset_count > 0:
                try:
                    import librosa as _librosa
                    onset_times_sec = _librosa.frames_to_time(
                        features.onset_frames, sr=audio.sample_rate, hop_length=512
                    )
                    onset_times_ms = [round(t * 1000, 1) for t in onset_times_sec]
                    context['onset_timestamps_ms'] = onset_times_ms[:20]
                    if len(onset_times_ms) > 20:
                        context['total_onsets'] = len(onset_times_ms)
                except Exception:
                    pass  # Non-critical

            # Chroma features (for key indication)
            chroma_mean = np.mean(features.chroma, axis=1)
            dominant_pitch_class = int(np.argmax(chroma_mean))
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            context['dominant_pitch'] = pitch_classes[dominant_pitch_class]

        except Exception as e:
            self.logger.warning(f"Could not extract all features: {e}")
            context['features_note'] = 'Some features could not be extracted'

        # Panning analysis (stereo only)
        if audio.channels == 2:
            try:
                context['panning'] = self._analyze_panning(audio)
            except Exception:
                context['panning'] = 'stereo (analysis unavailable)'
        else:
            context['panning'] = 'mono (no panning)'

        return context

    def _analyze_panning(self, audio: AudioSample) -> str:
        """Analyze stereo panning movement over time.

        Compares L/R channel energy in windowed segments.
        audio.audio_data shape is (2, samples) for stereo.
        """
        left = audio.audio_data[0]
        right = audio.audio_data[1]

        # Window size: ~50ms at sample_rate
        window_size = max(1, int(audio.sample_rate * 0.05))
        num_windows = len(left) // window_size

        if num_windows < 2:
            l_energy = float(np.sum(left ** 2))
            r_energy = float(np.sum(right ** 2))
            total = l_energy + r_energy
            if total == 0:
                return "silent"
            balance = (r_energy - l_energy) / total
            if abs(balance) < 0.1:
                return "centered"
            return "panned right" if balance > 0 else "panned left"

        # Check total energy first — if silent, skip windowed analysis
        total_energy = float(np.sum(left ** 2) + np.sum(right ** 2))
        if total_energy == 0:
            return "silent"

        balances = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            l_e = float(np.sum(left[start:end] ** 2))
            r_e = float(np.sum(right[start:end] ** 2))
            total = l_e + r_e
            balances.append((r_e - l_e) / total if total > 0 else 0.0)

        balances_arr = np.array(balances)
        mean_balance = float(np.mean(balances_arr))
        balance_range = float(np.max(balances_arr) - np.min(balances_arr))

        if balance_range > 0.3:
            mid = len(balances) // 2
            first_half = float(np.mean(balances_arr[:mid]))
            second_half = float(np.mean(balances_arr[mid:]))
            if second_half > first_half + 0.1:
                direction = "pans left-to-right"
            elif first_half > second_half + 0.1:
                direction = "pans right-to-left"
            else:
                direction = "oscillating/sweeping panning"
            return f"{direction} (range: {balance_range:.2f}, center: {mean_balance:+.2f})"
        else:
            if abs(mean_balance) < 0.1:
                return "centered (stereo)"
            elif mean_balance > 0.3:
                return "panned hard right"
            elif mean_balance > 0.1:
                return "panned slightly right"
            elif mean_balance < -0.3:
                return "panned hard left"
            else:
                return "panned slightly left"

    def _format_speech_context(self, context: Dict[str, Any]) -> str:
        """Format speech recognition context for the prompt."""
        if context.get('transcription') or context.get('speech_detected'):
            return (
                f"SPEECH RECOGNITION RESULTS: Speech detected. "
                f"Full transcription: '{context.get('transcription', '')}'. "
                f"Detected words: {', '.join(context.get('speech_words', [])) if context.get('speech_words') else 'N/A'}. "
                f"Language: {context.get('speech_language', 'unknown')}. "
                f"Confidence: {context.get('speech_confidence', 0.0):.2f}. "
                f"IMPORTANT: Use the transcription to understand content and context."
            )
        return "SPEECH RECOGNITION RESULTS: No speech detected."

    def _create_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create the analysis prompt for the LLM.

        Args:
            context: Audio characteristics

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this audio sample and provide a structured response.

AUDIO CHARACTERISTICS:
- File name: {context.get('file_name', 'unknown')}
- Duration: {context.get('duration', 'unknown')}
- Sample rate: {context.get('sample_rate', 'unknown')}
- Bit depth: {context.get('bit_depth', 'unknown')}
- Channels: {context.get('channels', 'unknown')}
- Format: {context.get('format', 'unknown')}
- Quality: {context.get('quality_tier', 'unknown')}
- Brightness: {context.get('brightness', 'unknown')}
- Texture: {context.get('texture', 'unknown')}
- Character: {context.get('character', 'unknown')}
- Complexity: {context.get('complexity', 'unknown')}
- Dominant pitch: {context.get('dominant_pitch', 'unknown')}
- Panning: {context.get('panning', 'unknown')}"""

        # Add temporal data if available
        if context.get('onset_timestamps_ms'):
            onsets = context['onset_timestamps_ms']
            prompt += f"\n- Onset timestamps (ms): {onsets}"
            if context.get('total_onsets'):
                prompt += f" (showing first 20 of {context['total_onsets']})"

        if context.get('beat_timestamps_ms'):
            beats = context['beat_timestamps_ms']
            prompt += f"\n- Beat timestamps (ms): {beats}"

        if context.get('tempo_bpm'):
            prompt += f"\n- Tempo: {context['tempo_bpm']} BPM"

        if context.get('is_one_shot') is not None:
            prompt += f"\n- One-shot: {'yes' if context['is_one_shot'] else 'no (loopable/rhythmic)'}"

        if context.get('num_beats'):
            prompt += f"\n- Number of beats: {context['num_beats']}"

        speech_context = self._format_speech_context(context)

        prompt += f"""

Based on these characteristics, provide a comprehensive analysis:

1. SUGGESTED NAME: A short, descriptive name for this audio sample (2-5 words, suitable for a file name or library entry). Be specific about what the sound IS, not just its qualities.

2. DESCRIPTION: Write a rich, vivid description (80-150 words) that covers ALL of the following:

   a) SOUND IDENTIFICATION: What specific sound source(s) are present? Be precise — not just "drum" but "punchy 808 kick with sub-bass tail" or "brushed jazz snare with room ambience". Identify specific sources: car engine, dog bark, vinyl crackle, analog synth pad, distorted guitar, field recording, etc.

   b) TEMPORAL STRUCTURE: Describe what happens over time using the onset/beat timestamps provided. Example: "Opens with a pronounced kick at 0ms, followed by a sharp hi-hat at 125ms, with a snare crack at 250ms." If beats are present, describe the beat pattern (e.g., "two-beat phrase with a kick on the downbeat and snare on the upbeat"). If it's a one-shot, describe the attack-sustain-decay shape.

   c) PANNING & SPATIAL: Describe the stereo positioning using the panning data. Does the sound pan left-to-right or right-to-left? Is it centered? Does it have stereo width, movement, or spatial depth?

   d) SONIC CHARACTER: Describe texture, timbre, tonal qualities, dynamic range, and any notable processing (reverb tail, distortion artifacts, filtering, compression pumping, bit-crushing, saturation).

   e) CREATIVE CONTEXT: What specific genre, mood, or production scenario fits this sample? Be imaginative and precise (e.g., "dark minimal techno breakdown" not "electronic music").

   CRITICAL RULES:
   - NEVER use generic filler: "versatile sound", "suitable for various genres", "adds texture to any mix"
   - NEVER start with "This" or "A" — begin with an evocative verb or adjective
   - Each description must be UNIQUE — vary sentence structure, vocabulary, and perspective
   - Reference specific millisecond time points when describing temporal events
   - Write as a world-class sound designer documenting a professional sample library

3. SPEECH DETECTION: Does this contain speech or vocals? If yes, what type?
   {speech_context}

4. TAGS: 5-10 specific tags (prefer "808-kick" over "drum", "granular-pad" over "synth", "lo-fi-texture" over "ambient")

5. CONFIDENCE: How confident are you in this analysis? (0.0 to 1.0)

CRITICAL: You MUST provide a description. Even if uncertain, provide your best analysis.
If speech transcription is provided, incorporate it into the description.

Respond in JSON format:
{{
    "suggested_name": "name here",
    "description": "description here (REQUIRED — 80-150 words, detailed, specific, creative)",
    "contains_speech": true/false,
    "detected_words": ["word1", "word2"] or null,
    "speech_language": "en" or null,
    "tags": ["specific-tag1", "specific-tag2", ...],
    "confidence": 0.8,
    "explanation": "brief explanation of analysis reasoning"
}}

IMPORTANT: Respond with ONLY valid JSON, no additional text."""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API.

        Delegates to the shared LLMClient when available, otherwise
        uses the internal OpenAI client directly.

        Args:
            prompt: The prompt to send

        Returns:
            The LLM response text
        """
        system_msg = (
            "You are an expert audio engineer and sound designer with deep "
            "knowledge of music production, sound design, and audio analysis. "
            "Provide creative, detailed, and evocative descriptions that capture "
            "the essence of the audio. Use professional terminology when "
            "appropriate. Always respond in valid JSON format."
        )

        if self._shared_client is not None:
            return self._shared_client.chat(
                system_prompt=system_msg,
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise AnalysisError(
                f"LLM API call failed: {e}",
                analyzer_name=self.name,
                original_error=str(e)
            )

    def _parse_response(self, response: str, speech_data: Optional[dict] = None, context: Optional[Dict[str, Any]] = None) -> LLMAnalysis:
        """
        Parse LLM response into LLMAnalysis.

        Args:
            response: JSON response from LLM
            speech_data: Optional speech recognition results from Whisper
            context: Optional audio context for fallback descriptions

        Returns:
            LLMAnalysis object (always returns a result, never None)
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith('```'):
                # Remove markdown code block
                lines = cleaned.split('\n')
                # Remove first and last lines if they're code block markers
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)

            data = json.loads(cleaned)

            # Get description - ensure it exists
            description = data.get('description', '').strip()
            if not description:
                description = 'No description provided'

            # Use Whisper results if available (more accurate)
            # Check if Whisper found speech OR transcription exists (even if contains_speech is False)
            whisper_has_transcription = speech_data and speech_data.get('transcription', '').strip()
            whisper_contains_speech = speech_data and speech_data.get('contains_speech', False)
            
            if whisper_has_transcription or whisper_contains_speech:
                # Use Whisper results (preferred - more accurate)
                contains_speech = whisper_contains_speech or bool(whisper_has_transcription)
                detected_words = speech_data.get('detected_words', [])
                speech_language = speech_data.get('speech_language')
                transcription = speech_data.get('transcription', '').strip() or None
                speech_confidence = speech_data.get('confidence', 0.0)
            else:
                # Fall back to LLM detection
                contains_speech = bool(data.get('contains_speech', False))
                detected_words = data.get('detected_words')
                speech_language = data.get('speech_language')
                transcription = None
                speech_confidence = None

            return LLMAnalysis(
                suggested_name=data.get('suggested_name', 'Unknown Audio'),
                name_confidence=float(data.get('confidence', 0.5)),
                description=description,
                contains_speech=contains_speech,
                detected_words=detected_words,
                speech_language=speech_language,
                transcription=transcription,
                speech_confidence=speech_confidence,
                tags=data.get('tags', []),
                model_used=f"{self.provider}/{self.model}",
                explanation=data.get('explanation')
            )

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return a basic result
            # Use Whisper results if available even if LLM parsing fails
            whisper_has_transcription = speech_data and speech_data.get('transcription', '').strip()
            whisper_contains_speech = speech_data and speech_data.get('contains_speech', False)
            
            if whisper_has_transcription or whisper_contains_speech:
                contains_speech = whisper_contains_speech or bool(whisper_has_transcription)
                detected_words = speech_data.get('detected_words', [])
                speech_language = speech_data.get('speech_language')
                transcription = speech_data.get('transcription', '').strip() or None
                speech_confidence = speech_data.get('confidence', 0.0)
            else:
                contains_speech = False
                detected_words = None
                speech_language = None
                transcription = None
                speech_confidence = None
            
            return LLMAnalysis(
                suggested_name="Unknown Audio",
                name_confidence=0.3,
                description=response[:500] if response else "Analysis failed",
                contains_speech=contains_speech,
                detected_words=detected_words,
                speech_language=speech_language,
                transcription=transcription,
                speech_confidence=speech_confidence,
                tags=["unclassified"],
                model_used=f"{self.provider}/{self.model}",
                explanation="Failed to parse structured response"
            )
