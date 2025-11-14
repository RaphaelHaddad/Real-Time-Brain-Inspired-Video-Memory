import cv2
import numpy as np
import base64
import asyncio
import httpx
import time
import json
import uuid
from tqdm.asyncio import tqdm
from typing import List, Dict, Any
from ..core.config import PipelineConfig
from ..core.metrics import MetricsTracker
from ..core.logger import get_logger
from ..core.platform import PlatformUtils

logger = get_logger(__name__)

class VLMExtractor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = MetricsTracker()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.device = PlatformUtils.get_device()

    async def process_video(self, video_path: str, output_path: str) -> str:
        """Main entry point for VLM extraction pipeline"""
        run_id = str(uuid.uuid4())
        logger.info(f"Starting VLM extraction with run ID: {run_id}")

        try:
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Calculate chunks
            chunk_size_frames = int(self.config.video.chunk_size_seconds * fps)
            total_chunks = max(1, int(total_frames / chunk_size_frames))

            results = []
            chunk_progress = tqdm(total=total_chunks, desc="Processing video chunks")

            for chunk_idx in range(total_chunks):
                start_frame = chunk_idx * chunk_size_frames
                end_frame = min((chunk_idx + 1) * chunk_size_frames, total_frames)

                # Extract frames for this chunk
                frames = await self._extract_frames(cap, start_frame, end_frame)
                if not frames:
                    continue

                # Convert to base64
                base64_frames = self._frames_to_base64(frames)

                # Get time range
                start_time = start_frame / fps
                end_time = end_frame / fps
                time_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}-{int(end_time//60):02d}:{int(end_time%60):02d}"

                # Call VLM API
                chunk_start = time.perf_counter()
                content = await self._call_vlm_api(base64_frames, chunk_idx)
                chunk_time = time.perf_counter() - chunk_start

                results.append({
                    "time": time_str,
                    "content": content,
                    "chunk_idx": chunk_idx,
                    "processing_time": chunk_time
                })

                self.metrics.record_timing(f"chunk_{chunk_idx}", "vlm_inference", chunk_time)
                chunk_progress.update(1)

            # Save results
            output_data = {
                "metadata": {
                    "run_id": run_id,
                    "video_path": video_path,
                    "total_chunks": total_chunks,
                    "config": self.config.dict()
                },
                "results": results
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"VLM extraction completed. Output saved to: {output_path}")
            self.metrics.save_metrics(f"metrics/vlm_{run_id}.json")
            return output_path

        finally:
            cap.release()
            await self.client.aclose()

    async def _extract_frames(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """Extract evenly spaced frames from video chunk"""
        frames = []
        num_frames_to_extract = min(self.config.video.frames_per_chunk, end_frame - start_frame)

        if num_frames_to_extract <= 0:
            return frames

        # Calculate frame indices to extract
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames_to_extract, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Optimize frame size for API calls
                if frame.shape[0] > 720:
                    scale = 720 / frame.shape[0]
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), 720))
                frames.append(frame)

        return frames

    def _frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert frames to base64-encoded JPEG strings"""
        base64_frames = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(f"data:image/jpeg;base64,{base64_str}")
        return base64_frames

    async def _call_vlm_api(self, base64_frames: List[str], chunk_idx: int) -> str:
        """Call VLM API with frames and get response"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.config.vlm.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.config.vlm.system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": frame_url,
                                            "detail": "auto"
                                        }
                                    } for frame_url in base64_frames
                                ],
                                {
                                    "type": "text",
                                    "text": self.config.vlm.user_prompt_template.format(chunk_idx=chunk_idx)
                                }
                            ]
                        }
                    ],
                    "temperature": self.config.vlm.temperature,
                    "top_p": self.config.vlm.top_p,
                    "max_tokens": self.config.vlm.max_tokens
                }

                headers = {
                    "Content-Type": "application/json"
                }
                
                # Only add Authorization header if API key is provided
                if self.config.vlm.api_key:
                    headers["Authorization"] = f"Bearer {self.config.vlm.api_key}"

                response = await self.client.post(
                    self.config.vlm.endpoint,
                    json=payload,
                    headers=headers
                )

                response.raise_for_status()
                response_json = response.json()

                return response_json["choices"][0]["message"]["content"]

            except Exception as e:
                logger.warning(f"VLM API attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                raise