from typing import Optional
import cv2
import time
import numpy as np


class CV2ReadVideo:
    def __init__(
        self,
        video_path: int,
        sampling_fps: Optional[int] = None
    ) -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_idx = 0
        self.sampling_rate = 1 if not sampling_fps else int(self.fps / sampling_fps)
        
        if self.sampling_rate != 1:
            self.fps = sampling_fps
        
        self.avg_fps = 0
        
    def frames(self):
        while True:
            has_frame, frame = self.cap.read()

            if has_frame:
                if self.frame_idx % self.sampling_rate == 0:
                    self._fps_start = time.perf_counter()
                    yield frame
                self.frame_idx += 1
            else:
                break
    
    def show(
        self,
        frame: np.ndarray,
        pause: int = 5,
        window_name: str = 'Preview Output',
        resize: bool = True,
        show_fps = True,
        show_avg_fps_over = 0
    ) -> None:
        if resize:
            if isinstance(resize, tuple):
                frame = cv2.resize(frame, resize)
            if isinstance(resize, int):
                frame = cv2.resize(frame, (resize, int(resize * (frame.shape[0] / frame.shape[1]))))
            else:
                frame = cv2.resize(frame, (1280, int(1280 * (frame.shape[0] / frame.shape[1]))))

        if show_avg_fps_over and self.frame_idx == 0:
            self.avg_fps_tic = time.time()
        
        if show_avg_fps_over and self.frame_idx % show_avg_fps_over == 0:
            self.avg_fps = show_avg_fps_over / (time.time() - self.avg_fps_tic)
            self.avg_fps_tic = time.time()
        
        if show_fps:
            cv2.putText(
                frame,
                f'Infernece FPS({1 / (time.perf_counter() - self._fps_start):.0f})',
                (20, 35),
                fontFace = 0,
                fontScale = 0.7,
                color = [200, 20, 0],
                thickness = 2
            )
            cv2.putText(
                frame,
                f'Sampling FPS({self.fps:.0f})',
                (20, 70),
                fontFace = 0,
                fontScale = 0.7,
                color = [240, 55, 155],
                thickness = 2
            )

            cv2.putText(
                frame,
                f'Average {show_avg_fps_over} FPS({self.avg_fps:.0f})',
                (20, 105),
                fontFace = 0,
                fontScale = 0.7,
                color = [240, 155, 20],
                thickness = 2
            )
            
        cv2.imshow(window_name, frame)
        # cv2.waitKey(int((1 / self.fps) * 1000))
        cv2.waitKey(pause)


class CV2WriteVideo:
    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        fps: int = 24
    ) -> None:
        assert 'mp4' in video_path, f'\n\nOnly mp4 format is supported for now.\n'

        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
            frameSize = [width, height],
            fps = fps
        )

        self.height = height
        self.width = width

    def write(self, frame):
        if frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self.video_writer.write(frame)
    
    def release(self):
        self.video_writer.release()
