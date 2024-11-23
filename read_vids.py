import imageio.v3 as iio
import os
from sys import argv
video_name = argv[1]

video = video_name
video_id = video.split("/")[-1].replace(".mp4","")


png_base = "png_logs"
try:
    os.mkdir(png_base)
except:
    pass

video_id = os.path.join(png_base, video_id)
all_frames = list(iio.imiter(video))

ctr = 0
try: 
    os.makedirs(video_id)
except:
    pass
for idx, frame in enumerate(all_frames):
    
    iio.imwrite(f"{video_id}/{ctr:03d}.jpg", frame)
    ctr += 1
