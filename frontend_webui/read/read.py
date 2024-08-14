
import os
import sys
import pyttsx3
import pygame.mixer
import time
def play_wav(text_file):
	text_file = text_file.replace('\"', '\'')
	print(f'text_file : {text_file}')
	if(os.path.exists(audio_path)):
	    pygame.mixer.music.load(audio_path)
	    pygame.mixer.music.play()
	start_time = time.time()
	cmd = f"echo \"{text_file}\" | piper   --model piper/en_US-kristin-medium.onnx  --output_file {audio_path}.backup "
	if(text_file != '' and text_file != '\n' and text_file.find('<pr>') == -1 and text_file.find('</pr>') == -1):
	    os.system(cmd)
	end_time = time.time()
	duration = end_time - start_time	
	print(f'exec time : {duration}')
	while pygame.mixer.music.get_busy():
	    continue
	if(os.path.exists(f'{audio_path}.backup')):
	    cmd = f"mv welcome.wav.backup welcome.wav"
	    os.system(cmd)
	else:
	    cmd = f"rm welcome.wav"
	    os.system(cmd)
	    

# cmd = f"echo {str} >> {file_path}"
# os.system(cmd)

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-30)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[14].id)
text_path = '/home/jichuang/workspace/tpu/cache/audio/audio.txt'
audio_path = 'welcome.wav'

def playaudio(text_path,last_position):
    with open(text_path, 'r') as f:
        text_lines = f.readlines()
    print(text_lines)
    for pos in range(last_position, len(text_lines)):
        # print(text_lines[pos])
        play_wav(text_lines[pos])
        # engine.say(text_lines[pos])
        # engine.runAndWait()
    play_wav('')
    return len(text_lines)

def start_service():
    import time
    
    last_modified = 0
    last_position = 0
    pygame.mixer.init()
    while True:
        if os.path.exists(text_path):
            modified_time = os.path.getmtime(text_path)
            # last_position = playaudio(text_path,last_position)
            if modified_time > last_modified:
                print('')
                last_modified = modified_time
                last_position = playaudio(text_path,last_position)
        time.sleep(0.5)
        sys.stdout.write('\rwait txt file')
        sys.stdout.flush()

if __name__ == '__main__':
    try:
       os.remove(text_path)
       print(f"Deleted {text_path}")
    except OSError as e:
       print(f"Error deleting {text_path}: {e}")
    try:
       os.remove(audio_path)
       print(f"Deleted {audio_path}")
    except OSError as e:
       print(f"Error deleting {audio_path}: {e}")
    try:
       os.remove(f'{audio_path}.backup')
       print(f"Deleted {audio_path}.backup")
    except OSError as e:
       print(f"Error deleting {audio_path}.backup: {e}")

    start_service()
