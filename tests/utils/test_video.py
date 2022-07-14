from triton_inference_clients.utils import CV2ReadVideo, CV2WriteVideo
import os

def test_video_reader():
    video_reader = CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'long_range_video (0).mp4'), sampling_fps = 2)
    
    assert video_reader.fps == 2
    assert video_reader.frame_idx == 0

    for frame in video_reader.frames():
        print(video_reader.frame_idx)

        # video_reader.show(frame, resize = (640, 580))
        if video_reader.frame_idx > 200:
            break

    assert video_reader.frame_idx > 0