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
        max_tokens: int = 1000,
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

    def _analyze_impl(self, audio: AudioSample, speech_data: Optional[dict] = None) -> LLMAnalysis:
        """
        Analyze audio using LLM.

        Args:
            audio: AudioSample to analyze
            speech_data: Optional speech recognition results from Whisper

        Returns:
            LLMAnalysis: LLM-generated analysis (always returns a result, never None)
        """
        # Build context from audio characteristics
        context = self._build_audio_context(audio)
        
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

            # Chroma features (for key indication)
            chroma_mean = np.mean(features.chroma, axis=1)
            dominant_pitch_class = int(np.argmax(chroma_mean))
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            context['dominant_pitch'] = pitch_classes[dominant_pitch_class]

        except Exception as e:
            self.logger.warning(f"Could not extract all features: {e}")
            context['features_note'] = 'Some features could not be extracted'

        return context

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

Based on these characteristics, provide a comprehensive analysis:

1. SUGGESTED NAME: A short, descriptive name for this audio sample (2-5 words, suitable for a file name or library entry)

2. DESCRIPTION: Write a rich, detailed description (aim for 30-50 words) that captures:
   - What the audio sounds like (sonic character, texture, timbre)
   - Its musical or sound design qualities
   - Potential uses (e.g., "drum fill", "ambient texture", "transition effect", "melodic element", "interview", "spoken word", "vocal sample")
   - Any notable characteristics (brightness, warmth, aggression, subtlety, etc.)
   - If speech is present: describe the content, tone, and context (e.g., "interview with energetic discussion", "calm narration", "aggressive spoken word")
   Be creative and specific - avoid generic phrases like "bright sound" or "percussive hit". Instead, describe it as a professional sound designer or audio archivist would.

3. SPEECH DETECTION: Does this appear to contain speech or vocals? If yes, what type (spoken word, singing, interview, narration, etc.)?
   {f"SPEECH RECOGNITION RESULTS: Speech detected. Full transcription: '{context.get('transcription', '')}'. Detected words: {', '.join(context.get('speech_words', [])) if context.get('speech_words') else 'N/A'}. Language: {context.get('speech_language', 'unknown')}. Confidence: {context.get('speech_confidence', 0.0):.2f}. IMPORTANT: Use the transcription to understand the content and context of the speech. For interviews, describe the topic and tone. For spoken word, describe the style and content." if (context.get('transcription') or context.get('speech_detected')) else "SPEECH RECOGNITION RESULTS: No speech detected by high-accuracy speech recognition system."}

4. TAGS: 5-10 relevant tags for categorization (e.g., "drum", "synth", "ambient", "one-shot", "loop", "fx", etc.)

5. CONFIDENCE: How confident are you in this analysis? (0.0 to 1.0)

CRITICAL: You MUST provide a description. Even if you're uncertain, provide your best analysis based on the available information.
If speech transcription is provided, incorporate it into the description to give context about what is being said.

Respond in JSON format:
{{
    "suggested_name": "name here",
    "description": "description here (REQUIRED - must be 30-50 words, detailed and specific)",
    "contains_speech": true/false,
    "detected_words": ["word1", "word2"] or null,
    "speech_language": "en" or null,
    "tags": ["tag1", "tag2", ...],
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
