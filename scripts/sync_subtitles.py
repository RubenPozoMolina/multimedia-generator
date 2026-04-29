
import os
import sys
import json
import logging
import whisper
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def align_subtitles(audio_path, subtitles_path, output_path, model_size="base"):
    logger.info(f"Loading Whisper model '{model_size}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)

    logger.info(f"Transcribing audio: {audio_path}")
    # We use verbose=False to keep logs clean
    result = model.transcribe(audio_path, verbose=False, language="es", word_timestamps=True)
    
    # Whisper segments have 'start', 'end', and 'text'
    segments = result.get("segments", [])
    
    # Flatten words for easier matching
    all_words = []
    for seg in segments:
        for word_info in seg.get('words', []):
            all_words.append(word_info)
    
    # for s in segments:
    #     logger.info(f"DEBUG: Segment: {s['start']:.2f} - {s['end']:.2f} : {s['text']}")
    
    logger.info(f"Loading original subtitles: {subtitles_path}")
    with open(subtitles_path, 'r', encoding='utf-8') as f:
        sub_data = json.load(f)
    
    original_subs = sub_data.get("subtitles", [])
    if not original_subs:
        logger.warning("No subtitles found in the JSON file.")
        return

    # Improved alignment strategy using flattened words
    new_subtitles = []
    
    logger.info("Aligning subtitles using word timestamps and phoneme-like matching...")
    
    current_word_idx = 0
    import re
    from difflib import SequenceMatcher

    # Pre-process all_words to be more matchable
    processed_words = []
    for w in all_words:
        cleaned = re.sub(r'[^\w]', '', w['word'].strip().lower())
        if cleaned:
            processed_words.append({'text': cleaned, 'start': w['start'], 'end': w['end']})

    for i, sub in enumerate(original_subs):
        sub_text = sub['text'].strip().lower()
        sub_words = [re.sub(r'[^\w]', '', w) for w in re.findall(r'\w+', sub_text)]
        sub_words = [w for w in sub_words if w]
        
        if not sub_words:
            new_subtitles.append(sub)
            continue
            
        first_word = sub_words[0]
        # Use first two words for better start anchor if available
        anchor_text = "".join(sub_words[:2])
        
        found = False
        # Search for anchor match
        for w_idx in range(current_word_idx, len(processed_words)):
            # Try matching with one or two words
            w_text = processed_words[w_idx]['text']
            w_text_2 = "".join([pw['text'] for pw in processed_words[w_idx:w_idx+2]])
            
            # Fuzzy match ratio
            ratio1 = SequenceMatcher(None, first_word, w_text).ratio()
            ratio2 = SequenceMatcher(None, anchor_text, w_text_2).ratio()
            
            if ratio1 > 0.7 or ratio2 > 0.7:
                # Found potential start
                start_time = processed_words[w_idx]['start']
                
                # Now search for the end of the subtitle text
                # Try to match the whole subtitle string in the next N words
                full_sub_clean = "".join(sub_words)
                best_end_time = processed_words[w_idx]['end']
                best_ratio = 0
                
                for length in range(1, min(len(sub_words) + 5, len(processed_words) - w_idx)):
                    candidate_text = "".join([pw['text'] for pw in processed_words[w_idx:w_idx+length]])
                    ratio = SequenceMatcher(None, full_sub_clean, candidate_text).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_end_time = processed_words[w_idx+length-1]['end']
                        current_word_idx = w_idx + length
                
                if best_ratio > 0.5:
                    new_sub = sub.copy()
                    new_sub['start'] = start_time
                    new_sub['end'] = best_end_time
                    new_subtitles.append(new_sub)
                    logger.info(f"Matched: '{sub['text']}' -> {start_time:.2f} - {best_end_time:.2f} (ratio: {best_ratio:.2f})")
                    found = True
                    break
        
        if not found:
            logger.warning(f"Could not find match for subtitle: '{sub['text']}'")
            new_subtitles.append(sub)

    sub_data['subtitles'] = new_subtitles
    
    logger.info(f"Saving aligned subtitles to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sync_subtitles.py <audio_path> <subtitles_path> [output_path]")
        sys.exit(1)
    
    audio_p = sys.argv[1]
    subs_p = sys.argv[2]
    out_p = sys.argv[3] if len(sys.argv) > 3 else subs_p.replace(".json", "_aligned.json")
    
    align_subtitles(audio_p, subs_p, out_p)
