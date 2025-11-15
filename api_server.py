#!/usr/bin/env python3
"""
FastAPI Server for roop-unleashed
Chạy ở chế độ public trên port 5349
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Optional, List, Dict
import os
import socket
import urllib.request
import json
import base64
import cv2
import numpy as np
import asyncio
from io import BytesIO
import time
import uuid
import shutil
from datetime import datetime
import subprocess

# Import các module từ roop core (không sửa core)
import roop.globals
import roop.metadata
from roop.core import decode_execution_providers, set_display_ui, live_swap
from settings import Settings
import roop.utilities as util
import ui.globals as uii

app = FastAPI(
    title=roop.metadata.name,
    version=roop.metadata.version,
    description="API Server for roop-unleashed - Real-time Face Swap"
)

# CORS middleware để cho phép truy cập từ bên ngoài
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (có thể giới hạn sau)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Tạo thư mục video_records để lưu video của users
VIDEO_RECORDS_DIR = os.path.join(os.path.dirname(__file__), "video_records")
os.makedirs(VIDEO_RECORDS_DIR, exist_ok=True)

# Mount video_records để public
app.mount("/video_records", StaticFiles(directory=VIDEO_RECORDS_DIR), name="video_records")

# Thư mục tạm để lưu video đang ghi
TEMP_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "temp_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# Dictionary để lưu VideoWriter và thông tin cho mỗi websocket connection
active_recordings: Dict[str, Dict] = {}  # {connection_id: {video_writer, user_id, temp_path, width, height, fps}}

# Dictionary để lưu stream rooms - SFU model tối ưu
stream_rooms: Dict[str, Dict] = {}  # {room_id: {created_at, user_id, connection_id, last_encoded_frame, viewers}}
# last_encoded_frame: bytes của frame đã encode sẵn (JPEG) - shared cho tất cả viewers
# viewers: list các viewer events (mỗi viewer có event riêng để signal frame mới)

count = 0
processed_frame = None


def decode_jwt_simple(token: str) -> Optional[Dict]:
    """
    Decode JWT token đơn giản (chỉ decode payload, không verify signature)
    Không cần PyJWT, chỉ decode base64
    """
    try:
        # JWT có format: header.payload.signature
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        # Decode payload (part 1)
        payload = parts[1]
        # Base64URL decode
        payload = payload.replace('-', '+').replace('_', '/')
        # Add padding if needed
        padding = len(payload) % 4
        if padding:
            payload += '=' * (4 - padding)
        
        decoded_bytes = base64.b64decode(payload)
        decoded_json = json.loads(decoded_bytes.decode('utf-8'))
        return decoded_json
    except Exception as e:
        print(f"Error decoding JWT: {e}")
        return None


@app.get("/")
async def root(request: Request):
    """Root endpoint - Serve web interface với kiểm tra token"""
    # Kiểm tra token từ query params
    token = request.query_params.get("token")
    
    if not token:
        # Không có token, redirect đến vtoobe.com
        return RedirectResponse(url="https://vtoobe.com", status_code=302)
    
    # Kiểm tra token hợp lệ
    decoded = decode_jwt_simple(token)
    if not decoded:
        # Token không hợp lệ, redirect đến vtoobe.com
        return RedirectResponse(url="https://vtoobe.com", status_code=302)
    
    user_id = decoded.get("user_id") or decoded.get("sub")
    if not user_id:
        # Không có user_id trong token, redirect đến vtoobe.com
        return RedirectResponse(url="https://vtoobe.com", status_code=302)
    
    # Token hợp lệ, serve web interface
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "name": roop.metadata.name,
        "version": roop.metadata.version,
        "status": "running",
        "endpoints": {
            "web_interface": "/",
            "health": "/health",
            "info": "/info",
            "api_docs": "/docs",
            "websocket": "/ws/video",
            "stop_recording": "/api/stop-recording"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": roop.metadata.name
    }


@app.get("/info")
async def get_info():
    """Thông tin về hệ thống"""
    return {
        "name": roop.metadata.name,
        "version": roop.metadata.version,
        "execution_providers": roop.globals.execution_providers if hasattr(roop.globals, 'execution_providers') else [],
        "config": {
            "server_port": 5349,
            "mode": "public"
        }
    }


@app.get("/api/status")
async def get_status():
    """Trạng thái xử lý"""
    return {
        "processing": getattr(roop.globals, 'processing', False),
        "execution_providers": getattr(roop.globals, 'execution_providers', []),
        "has_faces": len(getattr(roop.globals, 'INPUT_FACESETS', [])) > 0,
        "facesets_count": len(getattr(roop.globals, 'INPUT_FACESETS', []))
    }


@app.post("/api/upload-face")
async def upload_face(file: UploadFile = File(...)):
    """
    Upload face image và lưu vào source.png, sau đó reload face từ source.png
    """
    try:
        # Đọc file
        contents = await file.read()
        
        # Convert to numpy array để validate
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Lưu vào source.png
        source_path = os.path.join(os.path.dirname(__file__), "source.png")
        with open(source_path, 'wb') as f:
            f.write(contents)
        
        print(f'Face image saved to source.png')
        
        # Reload face từ source.png (sử dụng hàm có sẵn)
        success = load_source_face()
        
        if not success:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image. Please upload an image with a clear face.")
        
        facesets_count = len(getattr(roop.globals, 'INPUT_FACESETS', []))
        face_count = 0
        if facesets_count > 0 and len(roop.globals.INPUT_FACESETS[0].faces) > 0:
            face_count = len(roop.globals.INPUT_FACESETS[0].faces)
        
        return {
            "success": True,
            "message": f"Face uploaded and saved to source.png. {face_count} face(s) detected and loaded.",
            "facesets_count": facesets_count,
            "face_count": face_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing face: {str(e)}")


@app.delete("/api/clear-faces")
async def clear_faces():
    """Xóa tất cả faces đã upload"""
    if hasattr(roop.globals, 'INPUT_FACESETS'):
        roop.globals.INPUT_FACESETS = []
    return {"success": True, "message": "All faces cleared"}


@app.get("/view/{room_id}")
async def view_stream_page(room_id: str, request: Request):
    """
    Endpoint để hiển thị trang viewer với giao diện đẹp (không có nút share - share ở trang streamer)
    """
    # Kiểm tra room có tồn tại không
    if room_id not in stream_rooms:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stream Not Found - Vtoobe</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                    text-align: center; 
                    padding: 50px;
                    background: #f5f5f7;
                }
                .error { color: #ff3b30; font-size: 24px; margin-bottom: 16px; }
            </style>
        </head>
        <body>
            <div class="error">Stream not found or expired</div>
            <p>The stream room may have been closed or does not exist.</p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=404)
    
    # Lấy thông tin room
    room = stream_rooms[room_id]
    base_url = str(request.base_url).rstrip('/')
    share_url = f"{base_url}/view/{room_id}"
    
    # Tạo HTML page với giao diện đẹp
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Stream - Vtoobe</title>
        <meta property="og:title" content="Live Face Swap Stream - Vtoobe">
        <meta property="og:description" content="Watch real-time face swap stream">
        <meta property="og:image" content="{request.base_url}watch/{room_id}">
        <meta property="og:url" content="{share_url}">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Live Face Swap Stream - Vtoobe">
        <meta name="twitter:description" content="Watch real-time face swap stream">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            :root {{
                --apple-bg: #f5f5f7;
                --apple-card: #ffffff;
                --apple-text: #1d1d1f;
                --apple-text-secondary: #86868b;
                --apple-blue: #0071e3;
                --apple-blue-hover: #0077ed;
                --apple-border: #d2d2d7;
                --apple-success: #34c759;
                --apple-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                --apple-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                background: var(--apple-bg);
                color: var(--apple-text);
                min-height: 100vh;
                padding: 0;
                margin: 0;
                line-height: 1.5;
            }}
            
            .header {{
                background: var(--apple-card);
                border-bottom: 1px solid var(--apple-border);
                padding: 20px 0;
                text-align: center;
                position: sticky;
                top: 0;
                z-index: 100;
                backdrop-filter: blur(20px);
                background: rgba(255, 255, 255, 0.8);
            }}
            
            .logo {{
                font-size: 32px;
                font-weight: 600;
                letter-spacing: -0.5px;
                color: var(--apple-text);
                margin: 0;
            }}
            
            .tagline {{
                font-size: 14px;
                color: var(--apple-text-secondary);
                margin-top: 4px;
                font-weight: 400;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            
            .stream-container {{
                background: var(--apple-card);
                border-radius: 18px;
                padding: 30px;
                box-shadow: var(--apple-shadow);
                border: 1px solid var(--apple-border);
                margin-bottom: 24px;
            }}
            
            .stream-box {{
                position: relative;
                background: #000;
                border-radius: 18px;
                overflow: hidden;
                aspect-ratio: 16/9;
                box-shadow: var(--apple-shadow-lg);
                margin-bottom: 24px;
            }}
            
            .stream-box img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            
            .stream-status {{
                position: absolute;
                top: 16px;
                left: 16px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                backdrop-filter: blur(10px);
            }}
            
            .status-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--apple-success);
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            
            .share-section {{
                background: var(--apple-card);
                border-radius: 18px;
                padding: 30px;
                box-shadow: var(--apple-shadow);
                border: 1px solid var(--apple-border);
            }}
            
            .share-title {{
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 20px;
                color: var(--apple-text);
            }}
            
            .share-buttons {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                justify-content: center;
            }}
            
            .share-btn {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 24px;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 500;
                text-decoration: none;
                transition: all 0.2s ease;
                border: none;
                cursor: pointer;
                font-family: inherit;
            }}
            
            .share-btn:hover {{
                transform: translateY(-2px);
                box-shadow: var(--apple-shadow);
            }}
            
            .share-btn-facebook {{
                background: #1877f2;
                color: white;
            }}
            
            .share-btn-facebook:hover {{
                background: #166fe5;
            }}
            
            .share-btn-twitter {{
                background: #000000;
                color: white;
            }}
            
            .share-btn-twitter:hover {{
                background: #333333;
            }}
            
            .share-btn-tiktok {{
                background: #000000;
                color: white;
            }}
            
            .share-btn-tiktok:hover {{
                background: #333333;
            }}
            
            .share-btn-copy {{
                background: var(--apple-blue);
                color: white;
            }}
            
            .share-btn-copy:hover {{
                background: var(--apple-blue-hover);
            }}
            
            .copy-success {{
                display: none;
                margin-top: 12px;
                padding: 12px;
                background: rgba(52, 199, 89, 0.1);
                color: var(--apple-success);
                border-radius: 8px;
                font-size: 14px;
            }}
            
            .copy-success.show {{
                display: block;
            }}
            
            @media (max-width: 768px) {{
                .share-buttons {{
                    flex-direction: column;
                }}
                
                .share-btn {{
                    width: 100%;
                    justify-content: center;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 class="logo">Vtoobe</h1>
            <p class="tagline">Real-Time Face Swap Technology</p>
        </div>
        
        <div class="container">
            <div class="stream-container">
                <div class="stream-box">
                    <div class="stream-status">
                        <span class="status-dot"></span>
                        <span>LIVE</span>
                    </div>
                    <img id="streamImage" src="/watch/{room_id}" alt="Live Stream">
                </div>
            </div>
        </div>
        
        <script>
            // Handle stream image errors
            const streamImage = document.getElementById('streamImage');
            streamImage.onerror = function() {{
                this.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+U3RyZWFtIGVuZGVkPC90ZXh0Pjwvc3ZnPg==';
            }};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/watch/{room_id}")
async def watch_stream(room_id: str):
    """
    Endpoint để xem stream video sau khi swap face - Y HỆT STREAM GỐC
    Viewer nhận frame trực tiếp từ stream output, cùng quality và cơ chế như stream gốc
    """
    # Kiểm tra room có tồn tại không
    if room_id not in stream_rooms:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stream Not Found</title>
            <style>
                body { font-family: Arial; text-align: center; padding: 50px; }
                .error { color: #ff3b30; font-size: 24px; }
            </style>
        </head>
        <body>
            <div class="error">Stream not found or expired</div>
            <p>The stream room may have been closed or does not exist.</p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=404)
    
    # Tạo viewer event riêng để nhận signal frame mới (SFU tối ưu - không copy, không queue)
    room = stream_rooms[room_id]
    viewer_event = asyncio.Event()  # Mỗi viewer có event riêng
    
    # Thêm viewer vào room
    room["viewers"].append(viewer_event)
    viewer_count = len(room["viewers"])
    print(f"New viewer joined room {room_id}, total viewers: {viewer_count}")
    
    async def generate_frames():
        """
        Generator để tạo MJPEG stream - Y HỆT STREAM GỐC
        - Cùng quality 85 như stream gốc
        - Nhận frame real-time, không delay
        - Encode và gửi ngay lập tức
        """
        frame_count = 0
        
        try:
            while True:
                # Kiểm tra room còn tồn tại không
                if room_id not in stream_rooms:
                    print(f"Room {room_id} no longer exists, stopping stream")
                    break
                
                room_info = stream_rooms[room_id]
                
                # Kiểm tra room có bị đánh dấu xóa không
                if room_info.get("marked_for_deletion"):
                    deletion_time = room_info.get("deletion_time", 0)
                    if time.time() > deletion_time:
                        print(f"Room {room_id} deletion time reached, stopping stream")
                        break
                
                try:
                    # Đợi signal frame mới (real-time, không timeout)
                    # Frame được broadcast ngay sau khi swap (y hệt stream gốc)
                    await viewer_event.wait()
                    
                    # Lấy frame đã encode sẵn (shared, không copy, không encode lại)
                    room_info = stream_rooms[room_id]
                    frame_bytes = room_info.get("last_encoded_frame")
                    
                    # Reset event để đợi frame tiếp theo
                    viewer_event.clear()
                    
                    # Gửi frame ngay lập tức - Y HỆT STREAM GỐC (đã encode sẵn, không delay)
                    if frame_bytes is not None and len(frame_bytes) > 0:
                        # MJPEG format: boundary + frame (gửi ngay, không delay, không encode)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        frame_count += 1
                
                except asyncio.CancelledError:
                    # Connection closed
                    break
                except Exception as e:
                    print(f"Error in generate_frames loop for room {room_id}: {e}")
                    # Không sleep, tiếp tục đợi frame mới ngay
        except Exception as e:
            print(f"Error in generate_frames for room {room_id}: {e}")
        finally:
            # Xóa viewer khỏi room khi disconnect
            if room_id in stream_rooms:
                if viewer_event in stream_rooms[room_id]["viewers"]:
                    stream_rooms[room_id]["viewers"].remove(viewer_event)
                viewer_count = len(stream_rooms[room_id]["viewers"])
                print(f"Viewer left room {room_id}, remaining viewers: {viewer_count}")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


def get_video_duration(video_path: str) -> float:
    """
    Lấy duration của video file (tính bằng giây)
    Sử dụng OpenCV để đọc video
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        # Lấy FPS và frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            return duration
        return 0.0
    except Exception as e:
        print(f"Error getting video duration for {video_path}: {e}")
        return 0.0


@app.get("/api/total-duration")
async def get_total_duration(token: str = Query(..., description="JWT Token")):
    """
    Tính tổng thời lượng video của user từ folder video_records/{user_id}/
    """
    try:
        # Decode token để lấy user_id
        decoded = decode_jwt_simple(token)
        if not decoded:
            raise HTTPException(status_code=400, detail="Invalid token")
        
        user_id = decoded.get("user_id") or decoded.get("sub")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in token")
        
        # Đường dẫn folder của user
        user_folder = os.path.join(VIDEO_RECORDS_DIR, user_id)
        
        if not os.path.exists(user_folder):
            return {
                "success": True,
                "user_id": user_id,
                "total_duration_seconds": 0,
                "total_duration_formatted": "00:00:00",
                "video_count": 0
            }
        
        # Tính tổng thời lượng từ tất cả video files
        total_duration = 0.0
        video_count = 0
        
        for filename in os.listdir(user_folder):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                file_path = os.path.join(user_folder, filename)
                if os.path.isfile(file_path):
                    duration = get_video_duration(file_path)
                    total_duration += duration
                    video_count += 1
        
        # Format thời lượng thành HH:MM:SS
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return {
            "success": True,
            "user_id": user_id,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_formatted": formatted_duration,
            "video_count": video_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_total_duration: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting total duration: {str(e)}")


@app.get("/api/videos")
async def get_user_videos(token: str = Query(..., description="JWT Token")):
    """
    Lấy danh sách video records của user
    Nhận token JWT từ params, decode để lấy user_id
    Trả về danh sách video với đường dẫn, ngày tạo, tên file
    Sắp xếp từ mới đến cũ
    """
    try:
        # Decode token để lấy user_id
        decoded = decode_jwt_simple(token)
        if not decoded:
            raise HTTPException(status_code=400, detail="Invalid token")
        
        user_id = decoded.get("user_id") or decoded.get("sub")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in token")
        
        # Đường dẫn folder của user
        user_folder = os.path.join(VIDEO_RECORDS_DIR, user_id)
        
        if not os.path.exists(user_folder):
            return {
                "success": True,
                "user_id": user_id,
                "videos": [],
                "count": 0
            }
        
        # Lấy danh sách file video
        video_files = []
        for filename in os.listdir(user_folder):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                file_path = os.path.join(user_folder, filename)
                if os.path.isfile(file_path):
                    # Lấy thông tin file
                    file_stat = os.stat(file_path)
                    created_time = datetime.fromtimestamp(file_stat.st_ctime)
                    modified_time = datetime.fromtimestamp(file_stat.st_mtime)
                    file_size = file_stat.st_size
                    
                    # Tạo URL để truy cập video
                    video_url = f"/video_records/{user_id}/{filename}"
                    
                    video_files.append({
                        "filename": filename,
                        "url": video_url,
                        "created_at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "created_at_timestamp": int(file_stat.st_ctime),
                        "modified_at": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "modified_at_timestamp": int(file_stat.st_mtime),
                        "size": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    })
        
        # Sắp xếp từ mới đến cũ (theo created_at_timestamp)
        video_files.sort(key=lambda x: x["created_at_timestamp"], reverse=True)
        
        return {
            "success": True,
            "user_id": user_id,
            "videos": video_files,
            "count": len(video_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_videos: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting videos: {str(e)}")


@app.post("/api/stop-recording")
async def stop_recording(user_id: str = Query(..., description="User ID")):
    """
    Stop recording và lưu video vào folder của user
    Tìm file video tạm của user đó và move vào folder user
    """
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Tìm file video tạm của user này (tìm trong active_recordings theo user_id)
        temp_file = None
        conn_id_to_remove = None
        
        # Tìm trong active_recordings theo user_id
        for conn_id, recording_info in active_recordings.items():
            if recording_info.get("user_id") == user_id:
                temp_path = recording_info.get("temp_path")
                if temp_path and os.path.exists(temp_path):
                    temp_file = temp_path
                    conn_id_to_remove = conn_id
                    break
        
        # Nếu không tìm thấy trong active_recordings, tìm file mới nhất trong temp folder
        # (fallback cho trường hợp WebSocket đã disconnect nhưng file vẫn còn)
        if not temp_file:
            temp_files = [f for f in os.listdir(TEMP_VIDEO_DIR) if f.endswith('.avi')]
            if temp_files:
                # Lấy file mới nhất
                latest_time = 0
                for temp_file_name in temp_files:
                    temp_path = os.path.join(TEMP_VIDEO_DIR, temp_file_name)
                    file_time = os.path.getmtime(temp_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        temp_file = temp_path
        
        if not temp_file or not os.path.exists(temp_file):
            raise HTTPException(status_code=404, detail="No recording found for this user")
        
        # Đóng video writer nếu đang mở (từ active_recordings)
        if conn_id_to_remove and conn_id_to_remove in active_recordings:
            recording_info = active_recordings[conn_id_to_remove]
            video_writer = recording_info.get("video_writer")
            if video_writer:
                try:
                    video_writer.release()
                except:
                    pass
        
        # Tạo folder cho user nếu chưa có
        user_folder = os.path.join(VIDEO_RECORDS_DIR, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        # Tạo tên file video với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"video_{timestamp}.mp4"
        output_path = os.path.join(user_folder, output_filename)
        
        # Re-encode video với ffmpeg để đảm bảo tương thích
        print(f"Re-encoding video for user {user_id}...")
        try:
            cmd = [
                'ffmpeg',
                '-i', temp_file,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.isfile(output_path):
                # Xóa file tạm
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                # Xóa khỏi active_recordings nếu có
                if conn_id_to_remove and conn_id_to_remove in active_recordings:
                    del active_recordings[conn_id_to_remove]
                
                # Tạo URL để truy cập video
                video_url = f"/video_records/{user_id}/{output_filename}"
                
                print(f"Video saved for user {user_id}: {output_path}")
                return {
                    "success": True,
                    "message": "Video saved successfully",
                    "video_url": video_url,
                    "filename": output_filename
                }
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                raise Exception(f"FFmpeg encoding failed: {error_msg}")
        except Exception as e:
            print(f"Error re-encoding video: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in stop_recording: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket endpoint cho real-time video streaming và face swap
    Client sẽ gửi user_id trong message khi bắt đầu recording
    """
    await websocket.accept()
    
    # Tạo connection ID
    connection_id = str(uuid.uuid4())
    video_writer = None
    temp_video_path = None
    frame_width = None
    frame_height = None
    fps = 25  # Default FPS
    user_id = None  # Sẽ nhận từ client
    
    try:
        while True:
            # Nhận frame từ client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Nhận user_id từ client (gửi lần đầu khi bắt đầu)
            if message.get("type") == "start_recording":
                user_id = message.get("user_id")
                if user_id:
                    print(f"WebSocket connection - User ID: {user_id}")
                    
                    # Tạo room_id cho stream (SFU model tối ưu - không copy, encode một lần)
                    room_id = str(uuid.uuid4())
                    stream_rooms[room_id] = {
                        "created_at": datetime.now(),
                        "user_id": user_id,
                        "connection_id": connection_id,
                        "room_id": room_id,
                        "last_encoded_frame": None,  # JPEG bytes đã encode sẵn (shared)
                        "viewers": []  # List các viewer events (mỗi viewer có event riêng)
                    }
                    
                    # Lưu room_id vào active_recordings để dễ tìm
                    if connection_id in active_recordings:
                        active_recordings[connection_id]["room_id"] = room_id
                    
                    # Gửi room_id và watch link về client
                    await websocket.send_json({
                        "type": "room_created",
                        "room_id": room_id,
                        "watch_url": f"/view/{room_id}",  # Dùng /view thay vì /watch để có giao diện đẹp
                        "message": "Stream room created"
                    })
                    print(f"Created stream room: {room_id} for user: {user_id}")
            
            if message.get("type") == "frame":
                # Decode base64 frame
                frame_base64 = message.get("frame", "")
                timestamp = message.get("timestamp", 0)
                
                try:
                    # Decode base64 to image
                    frame_bytes = base64.b64decode(frame_base64)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to decode frame"
                        })
                        continue
                    
                    # Lấy kích thước frame lần đầu để khởi tạo video writer
                    if video_writer is None and connection_id not in active_recordings:
                        frame_height, frame_width = frame.shape[:2]
                        # Đảm bảo width và height là số chẵn
                        frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
                        frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1
                        
                        # Tạo file video tạm
                        temp_video_path = os.path.join(TEMP_VIDEO_DIR, f"{connection_id}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
                        
                        # Tìm room_id từ stream_rooms
                        room_id_for_recording = None
                        for rid, room_info in stream_rooms.items():
                            if room_info.get("connection_id") == connection_id:
                                room_id_for_recording = rid
                                break
                        
                        # Lưu thông tin recording
                        start_time = time.time()  # Thời gian bắt đầu recording
                        active_recordings[connection_id] = {
                            "video_writer": video_writer,
                            "user_id": user_id,
                            "temp_path": temp_video_path,
                            "width": frame_width,
                            "height": frame_height,
                            "fps": fps,
                            "room_id": room_id_for_recording,  # Lưu room_id vào đây
                            "start_time": start_time,  # Thời gian bắt đầu recording
                            "frame_count": 0  # Đếm số frame đã ghi
                        }
                        print(f"Started recording for connection {connection_id}, user: {user_id}, room: {room_id_for_recording}")
                    
                    # Thực hiện face swap sử dụng live_swap từ core
                    swapped_frame = process_frame_for_swap(frame)
                    
                    # Resize nếu cần
                    if video_writer and (swapped_frame.shape[1] != frame_width or swapped_frame.shape[0] != frame_height):
                        swapped_frame = cv2.resize(swapped_frame, (frame_width, frame_height))
                    
                    # Ghi frame vào video nếu đang recording
                    if video_writer and video_writer.isOpened():
                        video_writer.write(swapped_frame)
                        # Tăng frame count để tính thời lượng
                        if connection_id in active_recordings:
                            active_recordings[connection_id]["frame_count"] = active_recordings[connection_id].get("frame_count", 0) + 1
                    
                    # Lưu frame vào stream room để người khác xem
                    # Tìm room_id từ active_recordings hoặc từ stream_rooms
                    room_id_to_update = None
                    if connection_id in active_recordings:
                        room_id_to_update = active_recordings[connection_id].get("room_id")
                    
                    # Nếu không tìm thấy trong active_recordings, tìm trong stream_rooms
                    if not room_id_to_update:
                        for rid, room_info in stream_rooms.items():
                            if room_info.get("connection_id") == connection_id:
                                room_id_to_update = rid
                                break
                    
                    # Encode swapped frame một lần (dùng cho cả streamer và viewers)
                    _, buffer = cv2.imencode('.jpg', swapped_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()  # JPEG bytes
                    swapped_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                    
                    # Tính thời lượng video đã ghi (nếu đang recording)
                    recording_duration = 0
                    if connection_id in active_recordings:
                        recording_info = active_recordings[connection_id]
                        start_time = recording_info.get("start_time")
                        if start_time:
                            recording_duration = time.time() - start_time  # Thời lượng tính bằng giây
                    
                    # Kiểm tra giới hạn 1 giờ (3600 giây) cho free tier
                    # Tính tổng thời lượng đã ghi + thời lượng đang recording
                    total_duration_exceeded = False
                    if user_id:
                        try:
                            user_folder = os.path.join(VIDEO_RECORDS_DIR, user_id)
                            if os.path.exists(user_folder):
                                total_duration = 0.0
                                for filename in os.listdir(user_folder):
                                    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                                        file_path = os.path.join(user_folder, filename)
                                        if os.path.isfile(file_path):
                                            duration = get_video_duration(file_path)
                                            total_duration += duration
                                
                                # Thêm thời lượng đang recording
                                total_duration += recording_duration
                                
                                # Kiểm tra nếu vượt quá 1 giờ (3600 giây)
                                if total_duration >= 3600:
                                    total_duration_exceeded = True
                                    # Dừng recording nếu vượt quá giới hạn
                                    if video_writer and video_writer.isOpened():
                                        video_writer.release()
                                        if connection_id in active_recordings:
                                            active_recordings[connection_id]["video_writer"] = None
                        except Exception as e:
                            print(f"Error checking duration limit: {e}")
                    
                    # Gửi frame đã swap về client TRƯỚC (không bị block bởi broadcast)
                    await websocket.send_json({
                        "type": "frame",
                        "frame": swapped_base64,
                        "timestamp": timestamp,
                        "recording_duration": round(recording_duration, 1),  # Làm tròn 1 chữ số thập phân
                        "duration_limit_exceeded": total_duration_exceeded  # Thông báo nếu vượt quá giới hạn
                    })
                    
                    # Nếu vượt quá giới hạn, gửi cảnh báo
                    if total_duration_exceeded:
                        await websocket.send_json({
                            "type": "error",
                            "message": "You have reached the 1 hour recording limit. Recording stopped."
                        })
                    
                    # Broadcast frame đến viewers SAU (async, không block stream gốc)
                    # Tách ra để không làm chậm stream gốc
                    if room_id_to_update and room_id_to_update in stream_rooms:
                        # Chạy broadcast trong background task để không block
                        asyncio.create_task(broadcast_frame_to_viewers(room_id_to_update, frame_bytes))
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    })
            
            elif message.get("type") == "stop_recording":
                # Client yêu cầu stop recording
                break
                    
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Cleanup: đóng video writer và xóa khỏi active_recordings
        if connection_id in active_recordings:
            recording_info = active_recordings[connection_id]
            if recording_info["video_writer"]:
                recording_info["video_writer"].release()
            # Giữ lại thông tin để có thể lưu sau khi stop
            # Không xóa ngay, sẽ xóa khi call API stop-recording
            print(f"Stopped recording for connection {connection_id}")
        
        # Xóa stream room khi disconnect (SFU model)
        for room_id, room_info in list(stream_rooms.items()):
            if room_info.get("connection_id") == connection_id:
                viewers = room_info.get("viewers", [])
                viewers_count = len(viewers)
                if viewers_count == 0:
                    # Không còn người xem, xóa room ngay
                    del stream_rooms[room_id]
                    print(f"Removed stream room: {room_id} (no viewers)")
                else:
                    # Vẫn còn người xem, đánh dấu để xóa sau (sau 30 giây)
                    room_info["marked_for_deletion"] = True
                    room_info["deletion_time"] = time.time() + 30  # Xóa sau 30 giây
                    print(f"Marked room {room_id} for deletion (viewers: {viewers_count})")
                break


async def broadcast_frame_to_viewers(room_id: str, frame_bytes: bytes):
    """
    Broadcast frame đến viewers (async, không block stream gốc)
    """
    try:
        if room_id not in stream_rooms:
            return
        
        room_info = stream_rooms[room_id]
        viewers = room_info.get("viewers", [])
        
        if len(viewers) > 0:
            # Lưu frame đã encode sẵn (shared cho tất cả viewers)
            room_info["last_encoded_frame"] = frame_bytes
            
            # Signal tất cả viewers có frame mới (non-blocking, real-time)
            # Mỗi viewer có event riêng, signal tất cả cùng lúc
            for viewer_event in viewers:
                try:
                    viewer_event.set()  # Signal viewer này có frame mới
                except:
                    pass  # Viewer đã disconnect, bỏ qua
    except Exception as e:
        print(f"Error broadcasting frame for room {room_id}: {e}")


def process_frame_for_swap(frame):
    global count
    global processed_frame
    count = count + 1
    """
    Xử lý frame để swap face - sử dụng logic từ livecam_tab
    Không sửa core, chỉ gọi các hàm có sẵn
    """
    try:
        if(count % 6 == 0):
            start_time = time.time()
            # frame_height, frame_width = frame.shape[:2]
            # print(f"[Frame {count}] Input size: {frame_width}x{frame_height} | Channels: {frame.shape[2] if len(frame.shape) > 2 else 1}")
            
            # frame_copy = frame.copy()
            # frame_resized =  cv2.resize(frame_copy , (320, 240))
            
            # Kiểm tra xem có INPUT_FACESETS không
            if not hasattr(roop.globals, 'INPUT_FACESETS') or len(roop.globals.INPUT_FACESETS) == 0:
                print("Warning: No INPUT_FACESETS available, returning original frame")
                return frame
            
            # Đảm bảo TARGET_FACES được khởi tạo
            if not hasattr(roop.globals, 'TARGET_FACES'):
                roop.globals.TARGET_FACES = []

             # ⚠️ QUAN TRỌNG: Đảm bảo TẮT enhancer để tăng tốc
            roop.globals.selected_enhancer = None
            
            # Lấy các tham số từ globals (đã set trong initialize_roop)
            swap_mode = getattr(roop.globals, 'face_swap_mode', 'all')
            use_clip = False
            clip_text = None
            selected_index = getattr(uii, 'ui_SELECTED_INPUT_FACE_INDEX', 0)
            
            # Đảm bảo các tham số cần thiết được set
            if not hasattr(roop.globals, 'distance_threshold'):
                roop.globals.distance_threshold = 0.65
            if not hasattr(roop.globals, 'blend_ratio'):
                roop.globals.blend_ratio = 0.5
            
            # Gọi live_swap từ core (không sửa core, chỉ gọi hàm)
            # Sử dụng logic giống như on_stream_swap_cam trong livecam_tab.py
            swapped_frame = live_swap(frame, swap_mode, use_clip, clip_text, selected_index)
            
            # Nếu live_swap trả về None hoặc giống frame gốc, trả về frame đã swap
            if swapped_frame is None:
                print("Warning: live_swap returned None, returning original frame")
                return frame
            processed_frame = swapped_frame
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            return swapped_frame
        else:
            
            if(processed_frame is not None):
                return processed_frame
            else:
                return frame
        
    except Exception as e:
        print(f"Error in process_frame_for_swap: {e}")
        import traceback
        traceback.print_exc()
        # Nếu có lỗi, trả về frame gốc
        return frame
    

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Xử lý exception toàn cục"""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )


def get_public_ip():
    """Lấy public IP của server"""
    try:
        # Thử nhiều API đ ể lấy publicIP
        services = [
            'https://api.ipify.org?format=json',
            'https://ifconfig.me/ip',
            'https://icanhazip.com',
            'https://checkip.amazonaws.com'
        ]
        
        for service in services:
            try:
                with urllib.request.urlopen(service, timeout=3) as response:
                    if 'json' in service:
                        data = json.loads(response.read().decode())
                        return data.get('ip', '')
                    else:
                        return response.read().decode().strip()
            except:
                continue
        
        # Fallback: lấy local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except:
            local_ip = '127.0.0.1'
        finally:
            s.close()
        
        return local_ip
    except Exception as e:
        return None


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


def load_source_face():
    """
    Tự động load face từ source.png nếu file tồn tại
    Không sửa core, chỉ gọi các hàm có sẵn
    """
    source_path = os.path.join(os.path.dirname(__file__), "source.png")
    
    if not os.path.exists(source_path):
        print(f'Source image not found: {source_path}')
        return False
    
    try:
        from roop.face_util import extract_face_images
        from roop.FaceSet import FaceSet
        
        # Initialize INPUT_FACESETS nếu chưa có
        if not hasattr(roop.globals, 'INPUT_FACESETS'):
            roop.globals.INPUT_FACESETS = []
        
        # Clear existing facesets để load face mới từ source.png
        roop.globals.INPUT_FACESETS = []
        
        print(f'Loading face from source.png...')
        
        # Extract faces từ source.png
        face_data = extract_face_images(source_path, (False, 0))
        
        if len(face_data) == 0:
            print(f'No face detected in source.png')
            return False
        
        # Tạo FaceSet
        face_set = FaceSet()
        source_image = cv2.imread(source_path)
        
        for f in face_data:
            face = f[0]
            face.mask_offsets = (0, 0)
            face_set.faces.append(face)
            face_set.ref_images.append(source_image)
        
        # Nếu có nhiều faces, tính average embeddings
        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        
        # Thêm vào INPUT_FACESETS (thay thế nếu đã có)
        roop.globals.INPUT_FACESETS = [face_set]
        
        print(f'Successfully loaded {len(face_set.faces)} face(s) from source.png')
        return True
        
    except Exception as e:
        print(f'Error loading source.png: {e}')
        return False


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
    roop.globals.distance_threshold = 0.65  # Ngưỡng khoảng cách để detect face
    roop.globals.blend_ratio = 0.5  # Tỷ lệ blend
    roop.globals.selected_enhancer = None  # Không dùng enhancer mặc định (có thể set sau)
    
    # Tự động load face từ source.png
    load_source_face()
    
    print(f'FastAPI Server initialized')
    print(f'Using provider {roop.globals.execution_providers}')
    print(f'INPUT_FACESETS count: {len(roop.globals.INPUT_FACESETS)}')
    print(f'Face swap mode: {roop.globals.face_swap_mode}')
    print(f'Distance threshold: {roop.globals.distance_threshold}')
    print(f'Blend ratio: {roop.globals.blend_ratio}')


def run_server():
    """Chạy FastAPI server ở chế độ public"""
    # Khởi tạo roop core
    initialize_roop()
    
    # Lấy IP addresses
    public_ip = get_public_ip()
    local_ip = get_local_ip()
    
    # Chạy server với uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",  # Public mode - listen on all interfaces
        port=5349,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    print(f"\n{'='*60}")
    print(f"FastAPI Server starting...")
    print(f"{'='*60}")
    print(f"Server (Public):  http://0.0.0.0:5349")
    if public_ip:
        print(f"Public IP:       http://{public_ip}:5349")
    print(f"Local IP:        http://{local_ip}:5349")
    print(f"Localhost:       http://127.0.0.1:5349")
    print(f"{'='*60}")
    print(f"Web Interface:")
    if public_ip:
        print(f"  Real-Time Face Swap: http://{public_ip}:5349/")
    print(f"  Local Access:        http://{local_ip}:5349/")
    print(f"{'='*60}")
    print(f"API Documentation:")
    if public_ip:
        print(f"  Swagger UI:    http://{public_ip}:5349/docs")
    print(f"  ReDoc:         http://{public_ip if public_ip else local_ip}:5349/redoc")
    print(f"{'='*60}\n")
    
    server.run()


if __name__ == "__main__":
    run_server()

