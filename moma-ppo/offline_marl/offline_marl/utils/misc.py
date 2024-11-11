import numpy as np
import datetime

def save_video_to_wandb(wandb, title, frames):
    if not frames is None:
        wandb.log({title: wandb.Video(np.moveaxis(np.asarray(frames), [3], [1]), fps=24, format="mp4")})
    
def get_timestamp():
    dt = datetime.datetime.now()
    dt = str(dt.day) + '-' + str(dt.month) + '-' + str(dt.year) + '_' + str(dt.hour) + '_' + str(dt.minute)
    return dt