import subprocess
input_file = " "
output_file = " "
subprocess.run(["ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_file])