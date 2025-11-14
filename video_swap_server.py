#!/usr/bin/env python3
"""
FastAPI Server for Video Face Swap
Nhận ảnh và video, swap mặt, trả về link tải
Chạy ở port 5349
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import socket
import cv2
import numpy as np
import uuid
import shutil

# Import các module từ roop core (không sửa core)
import roop.globals
import roop.metadata
from roop.core import decode_execution_providers
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.face_util import extract_face_images
from roop.FaceSet import FaceSet
from roop.utilities import detect_fps, is_video, has_image_extension
from roop.capturer import get_video_frame_total
from settings import Settings
import roop.utilities as util

app = FastAPI(
    title="Video Face Swap API",
    version="1.0.0",
    description="API Server for Video Face Swap - Upload image and video, get swapped video download link"
)

# CORS middleware để cho phép truy cập từ bên ngoài
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục videos_output nếu chưa có
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "videos_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thư mục tạm để lưu file upload
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thư mục để lưu frames đã xử lý (public để xem)
FRAMES_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "frames_output")
os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
# if os.path.exists(static_dir):
#     app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount frames output directory để có thể xem ảnh
# if os.path.exists(FRAMES_OUTPUT_DIR):
#     app.mount("/frames", StaticFiles(directory=FRAMES_OUTPUT_DIR), name="frames")

# if os.path.exists(FRAMES_OUTPUT_DIR):
app.mount("/videos_output", StaticFiles(directory=OUTPUT_DIR), name="videos_output")


def get_processing_plugins(use_clip):
    """Lấy danh sách plugins cần thiết"""
    processors = "faceswap"
    if use_clip:
        processors += ",mask_clip2seg"
    
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors += ",dmdnet"
    elif roop.globals.selected_enhancer == 'GPEN':
        processors += ",gpen"
    return processors


def load_face_from_image(image_path: str):
    """
    Load face từ ảnh và thêm vào INPUT_FACESETS
    Không sửa core, chỉ gọi các hàm có sẵn
    """
    try:
        # Initialize INPUT_FACESETS nếu chưa có
        if not hasattr(roop.globals, 'INPUT_FACESETS'):
            roop.globals.INPUT_FACESETS = []
        
        # Clear existing facesets để load face mới
        roop.globals.INPUT_FACESETS = []
        
        print(f'Loading face from {image_path}...')
        
        # Extract faces từ image
        face_data = extract_face_images(image_path, (False, 0))
        
        if len(face_data) == 0:
            print(f'No face detected in image')
            return False
        
        # Tạo FaceSet
        face_set = FaceSet()
        source_image = cv2.imread(image_path)
        
        for f in face_data:
            face = f[0]
            face.mask_offsets = (0, 0)
            face_set.faces.append(face)
            face_set.ref_images.append(source_image)
        
        # Nếu có nhiều faces, tính average embeddings
        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        
        # Thêm vào INPUT_FACESETS
        roop.globals.INPUT_FACESETS = [face_set]
        
        print(f'Successfully loaded {len(face_set.faces)} face(s) from image')
        return True
        
    except Exception as e:
        print(f'Error loading face from image: {e}')
        import traceback
        traceback.print_exc()
        return False


def process_video_swap(video_path: str, output_path: str, request_id: str):
    """
    Xử lý video với face swap bằng OpenCV
    Đọc từng frame, swap face, lưu frame vào folder, tạo video bằng OpenCV
    """
    try:
        # Đảm bảo TARGET_FACES được khởi tạo
        if not hasattr(roop.globals, 'TARGET_FACES'):
            roop.globals.TARGET_FACES = []
        
        # Kiểm tra có INPUT_FACESETS không
        if not hasattr(roop.globals, 'INPUT_FACESETS') or len(roop.globals.INPUT_FACESETS) == 0:
            raise Exception("No face loaded. Please upload a face image first.")
        
        # Tạo ProcessMgr để swap face
        process_mgr = ProcessMgr(None)
        
        # Set options
        use_clip = False
        clip_text = None
        options = ProcessOptions(
            get_processing_plugins(use_clip),
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            0,
            clip_text
        )
        
        process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
        
        # Mở video input
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Đảm bảo width và height là số chẵn (yêu cầu của nhiều codec)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        print(f'Processing video: {video_path}')
        print(f'Output: {output_path}')
        print(f'FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}')
        
        # Tạo folder để lưu frames đã xử lý
        frames_folder = os.path.join(FRAMES_OUTPUT_DIR, request_id)
        os.makedirs(frames_folder, exist_ok=True)
        
        # Tạo VideoWriter với OpenCV - dùng codec XVID hoặc mp4v
        # Sẽ re-encode sau với ffmpeg để đảm bảo tương thích
        temp_video_path = output_path + ".temp_opencv.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec tương thích tốt hơn
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Swap face trên frame
            processed_frame = process_mgr.process_frame(frame)
            if processed_frame is None:
                processed_frame = frame
            
            # Resize nếu cần (đảm bảo đúng kích thước)
            if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                processed_frame = cv2.resize(processed_frame, (width, height))
            
            # Lưu frame đã xử lý vào folder
            frame_filename = os.path.join(frames_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, processed_frame)
            
            # Ghi frame vào video
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f'Processed {frame_count}/{total_frames} frames')
        
        # Release resources
        cap.release()
        out.release()
        process_mgr.release_resources()
        
        print(f'Processed {frame_count} frames')
        print(f'Frames saved to: {frames_folder}')
        
        # Re-encode video với ffmpeg từ temp file sang output
        if os.path.isfile(temp_video_path):
            print(f'Re-encoding video with ffmpeg for compatibility...')
            try:
                import subprocess
                # Re-encode với libx264 để đảm bảo tương thích
                cmd = [
                    'ffmpeg',
                    '-i', temp_video_path,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-y',
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Xóa file temp
                try:
                    os.remove(temp_video_path)
                except:
                    pass
                
                if result.returncode == 0 and os.path.isfile(output_path):
                    print(f'Video re-encoded successfully')
                else:
                    print(f'Warning: Re-encoding failed')
                    if result.stderr:
                        print(f'FFmpeg error: {result.stderr}')
            except Exception as e:
                print(f'Warning: Re-encoding error: {e}')
        
        if os.path.isfile(output_path):
            print(f'Video processed successfully: {output_path}')
            return True, frames_folder
        else:
            print(f'Failed to create output video')
            return False, None
            
    except Exception as e:
        print(f'Error processing video: {e}')
        import traceback
        traceback.print_exc()
        return False, None


def initialize_roop():
    """Khởi tạo roop core (không sửa core, chỉ gọi hàm có sẵn)"""
    # Load settings
    if not hasattr(roop.globals, 'CFG') or roop.globals.CFG is None:
        roop.globals.CFG = Settings('config.yaml')
    
    # Decode execution providers
    if not hasattr(roop.globals, 'execution_providers'):
        roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    
    # Set execution threads
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    
    # Initialize INPUT_FACESETS và TARGET_FACES
    if not hasattr(roop.globals, 'INPUT_FACESETS'):
        roop.globals.INPUT_FACESETS = []
    if not hasattr(roop.globals, 'TARGET_FACES'):
        roop.globals.TARGET_FACES = []
    
    # Set các tham số cần thiết cho face swap (mặc định)
    roop.globals.face_swap_mode = 'all'  # Swap tất cả faces
    roop.globals.distance_threshold = 0.65
    roop.globals.blend_ratio = 0.5
    roop.globals.selected_enhancer = None  # Không dùng enhancer mặc định
    roop.globals.skip_audio = True  # Bỏ audio, không restore
    
    # Set video encoder và quality
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    
    print(f'Video Swap Server initialized')
    print(f'Using provider {roop.globals.execution_providers}')
    print(f'Execution threads: {roop.globals.execution_threads}')


@app.get("/")
async def root():
    """Root endpoint - Serve web interface"""
    html_path = os.path.join(static_dir, "video_swap.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return {
        "name": "Video Face Swap API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "web_interface": "/",
            "upload_and_process": "/api/process-video",
            "download": "/api/download/{video_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Video Face Swap API"
    }


@app.post("/api/process-video")
async def process_video(
    face_image: UploadFile = File(...),
    video: UploadFile = File(...)
):
    """
    Nhận ảnh và video, swap mặt trong video, trả về link tải
    """
    try:
        # Validate file types
        if not has_image_extension(face_image.filename):
            raise HTTPException(status_code=400, detail="Face image must be PNG, JPG, or JPEG")
        
        if not is_video(video.filename) and not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Video must be a valid video file")
        
        # Tạo ID duy nhất cho request này
        request_id = str(uuid.uuid4())
        
        # Lưu file tạm
        face_image_path = os.path.join(UPLOAD_DIR, f"{request_id}_face_{face_image.filename}")
        video_path = os.path.join(UPLOAD_DIR, f"{request_id}_video_{video.filename}")
        
        # Lưu face image
        with open(face_image_path, 'wb') as f:
            contents = await face_image.read()
            f.write(contents)
        
        # Validate face image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            os.remove(face_image_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Lưu video
        with open(video_path, 'wb') as f:
            video_contents = await video.read()
            f.write(video_contents)
        
        # Load face từ image
        success = load_face_from_image(face_image_path)
        if not success:
            os.remove(face_image_path)
            os.remove(video_path)
            raise HTTPException(
                status_code=400,
                detail="No face detected in uploaded image. Please upload an image with a clear face."
            )
        
        # Tạo output path
        video_ext = os.path.splitext(video.filename)[1] or '.mp4'
        output_filename = f"{request_id}_swapped{video_ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Xử lý video
        print(f"Starting video processing for request {request_id}")
        success, frames_folder = process_video_swap(video_path, output_path, request_id)
        
        # Xóa file tạm
        try:
            os.remove(face_image_path)
            os.remove(video_path)
        except:
            pass
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video")
        
        # Trả về link tải và link xem frames
        download_url = f"/api/download/{request_id}"
        frames_url = f"/frames/{request_id}" if frames_folder else None
        
        return {
            "success": True,
            "message": "Video processed successfully",
            "video_id": request_id,
            "download_url": download_url,
            "filename": output_filename,
            "frames_url": frames_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in process_video: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/api/download/{video_id}")
async def download_video(video_id: str):
    """
    Tải video đã xử lý
    """
    try:
        # Tìm file video với video_id
        video_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{video_id}_swapped")]
        
        if not video_files:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = os.path.join(OUTPUT_DIR, video_files[0])
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_files[0],
            headers={"Content-Disposition": f"attachment; filename={video_files[0]}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Xử lý exception toàn cục"""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )


def get_local_ip():
    """Lấy local IP của server"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except:
            local_ip = '127.0.0.1'
        finally:
            s.close()
        return local_ip
    except:
        return '127.0.0.1'


def run_server():
    """Chạy FastAPI server"""
    # Khởi tạo roop core
    initialize_roop()
    
    # Lấy IP addresses
    local_ip = get_local_ip()
    
    # Chạy server với uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=5349,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    print(f"\n{'='*60}")
    print(f"Video Face Swap Server starting...")
    print(f"{'='*60}")
    print(f"Server (Public):  http://0.0.0.0:5349")
    print(f"Local IP:        http://{local_ip}:5349")
    print(f"Localhost:       http://127.0.0.1:5349")
    print(f"{'='*60}")
    print(f"API Endpoints:")
    print(f"  Upload & Process: POST http://{local_ip}:5349/api/process-video")
    print(f"  Download Video:   GET  http://{local_ip}:5349/api/download/{{video_id}}")
    print(f"  Health Check:      GET  http://{local_ip}:5349/health")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    server.run()


if __name__ == "__main__":
    run_server()

