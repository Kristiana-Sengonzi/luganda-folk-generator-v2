def extract_pure_lyrics(raw_text):
    """Extract only the content between 'Start' and explanatory text or song sections"""
    lines = raw_text.split('\n')
    song_lines = []
    found_start = False
    
    for line in lines:
        line = line.strip()
        
        # Start capturing when we find "Start" OR the first song section
        if not found_start and (line == "Start" or 
                               (line.startswith('[') and any(section in line for section in ['Verse', 'Chorus', 'Call', 'Response']))):
            found_start = True
            # If we found "Start", don't include that line - start from next line
            if line == "Start":
                continue
            
        if found_start:
            # Stop if we hit explanatory text
            if any(bad in line.lower() for bad in ['(note:', 'example:', 'i incorporated', 'explanation', 
                                                  'this version', 'as written', 'complete folksong', 
                                                  'preserves the', 'indicating its completion']):
                break
            # Also stop if we find "End" markers (but include the line if it's part of lyrics)
            if line.lower() in ['end', '[end]'] and not any(section in line for section in ['[Verse]', '[Chorus]', '[Call]', '[Response]']):
                break
            song_lines.append(line)
    
    # Remove any empty lines at the end
    while song_lines and not song_lines[-1].strip():
        song_lines.pop()
        
    return '\n'.join(song_lines)




