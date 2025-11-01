import json
import os
from app.audio_features_utils import get_audio_features

class LyricEmotionExtractor:
    def __init__(self):
        # Load Hartmann emotion model
        self.emotion_classifier = pipeline(
            "text-classification",
            model="/kaggle/input/j-hartmann/j hartmann",
            device=-1 
        )
        
        # Emotion mapping: Hartmann → Our emotions
        self.emotion_map = {
            'joy': 'joyful_activation',
            'sadness': 'sadness',
            'anger': 'tension',
            'fear': 'tension',
            'disgust': 'solemnity',
            'surprise': 'amazement',
            'neutral': 'calmness'
        }
        
        # Folk singing timing adjustments
        self.WORDS_PER_MINUTE = 100
        self.SECONDS_PER_WORD = 60 / self.WORDS_PER_MINUTE
        
        self.EMOTION_TIMING = {
            'sadness': 1.3,        # 30% slower - melancholic
            'solemnity': 1.25,     # 25% slower - reverent
            'calmness': 1.2,       # 20% slower - peaceful
            'tension': 1.15,       # 15% faster - urgent
            'amazement': 1.1,      # 10% faster - excited
            'joyful_activation': 1.0,  # Normal pace
            'default': 1.2
        }
        
        print(" Lyric Emotion Extractor initialized!")
    
    def clean_lyrics(self, lyrics):
        """Split lyrics into lines and clean them"""
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        # Remove empty lines and very short lines (like section headers)
        lines = [line for line in lines if len(line.split()) >= 2]
        return lines
    
    def classify_line_emotion(self, text_line):
        """Classify a single line's emotion using Hartmann model"""
        result = self.emotion_classifier(text_line)[0]
        
        hartmann_emotion = result['label']
        confidence = result['score']
        
        # Map to our emotion system
        our_emotion = self.emotion_map.get(hartmann_emotion, 'calmness')
        
        return {
            'hartmann_emotion': hartmann_emotion,
            'mapped_emotion': our_emotion,
            'confidence': confidence
        }
    
    def estimate_duration(self, text_line, emotion):
        """Estimate singing duration based on word count and emotion"""
        word_count = len(text_line.split())
        base_duration = word_count * self.SECONDS_PER_WORD
        
        # Get emotion-specific timing
        timing_multiplier = self.EMOTION_TIMING.get(emotion, self.EMOTION_TIMING['default'])
        duration = base_duration * timing_multiplier
        
        return round(duration, 2)
    
    def process_lyrics(self, lyrics):
        """Main function: Process complete lyrics"""
        lines = self.clean_lyrics(lyrics)
        processed_lines = []
        
        print(f" Processing {len(lines)} lyric lines...")
        
        for i, line in enumerate(lines):
            print(f"   Analyzing line {i+1}/{len(lines)}: '{line}'")
            
            # Step 1: Classify emotion
            emotion_result = self.classify_line_emotion(line)
            
            # Step 2: Estimate duration
            duration = self.estimate_duration(line, emotion_result['mapped_emotion'])
            
            # Step 3: Get audio features using YOUR existing function
            audio_features = get_audio_features(emotion_result['mapped_emotion'])
            
            processed_line = {
                'line_number': i + 1,
                'text': line,
                'hartmann_emotion': emotion_result['hartmann_emotion'],
                'mapped_emotion': emotion_result['mapped_emotion'],
                'confidence': emotion_result['confidence'],
                'duration_seconds': duration,
                'audio_features': audio_features,
                'word_count': len(line.split())
            }
            
            processed_lines.append(processed_line)
        
        return processed_lines
    
    def print_summary(self, processed_lyrics):
        """Print a summary of the processed lyrics"""
        print("\n" + "="*60)
        print("LYRIC EMOTION ANALYSIS SUMMARY")
        print("="*60)
        
        total_duration = sum(line['duration_seconds'] for line in processed_lyrics)
        emotion_counts = {}
        
        for line in processed_lyrics:
            emotion = line['mapped_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"Total lines: {len(processed_lyrics)}")
        print(f"Total estimated duration: {total_duration:.1f} seconds")
        print(f"Emotion distribution:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(processed_lyrics)) * 100
            print(f"  {emotion}: {count} lines ({percentage:.1f}%)")
        
        print("\nLine-by-line breakdown:")
        for line in processed_lyrics:
            print(f"  Line {line['line_number']}: '{line['text']}'")
            print(f"    Emotion: {line['mapped_emotion']} (from {line['hartmann_emotion']})")
            print(f"    Duration: {line['duration_seconds']}s, Words: {line['word_count']}")
            print()

    def get_audio_arrays(self, processed_lyrics):
        """Return separate arrays for duration, tempo, and energy"""
        durations = []
        tempos = []
        energies = []
        
        for line in processed_lyrics:
            # Safely get duration with default
            duration = line.get('duration_seconds', 2.0)  # default 2 seconds
            durations.append(duration)
            
            # Safely get audio features
            audio_features = line.get('audio_features', {})
            if audio_features:
                tempo = audio_features.get('tempo', 120.0)
                energy = audio_features.get('rms_energy', 0.08)
            else:
                tempo = 120.0  # default tempo
                energy = 0.08  # default energy
                
            tempos.append(tempo)
            energies.append(energy)
        
        return  durations, tempos, energies