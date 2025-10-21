#!/usr/bin/env python3
"""
DeepSeek OCR Web Server

A Flask-based web server that provides a simple API and web interface
for uploading and processing images with DeepSeek OCR.

Usage:
    python web_server.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
from datetime import datetime

# Add current directory to path for batch_ocr import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_ocr import DeepSeekOCRWrapper

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global job storage
jobs = {}
job_lock = threading.Lock()

# Initialize OCR wrapper (lazy loading)
ocr_wrapper = None
ocr_lock = threading.Lock()


def get_ocr_wrapper(backend='vllm', mode='gundam'):
    """Get or initialize OCR wrapper"""
    global ocr_wrapper
    with ocr_lock:
        if ocr_wrapper is None:
            print(f"Initializing OCR wrapper with backend={backend}, mode={mode}")
            ocr_wrapper = DeepSeekOCRWrapper(
                backend=backend,
                mode=mode,
                gpu_id="0"
            )
    return ocr_wrapper


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_job(job_id, image_paths, prompt, backend, mode):
    """Process OCR job in background"""
    try:
        with job_lock:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['started_at'] = datetime.now().isoformat()

        # Get OCR wrapper
        wrapper = get_ocr_wrapper(backend=backend, mode=mode)

        # Create output directory for this job
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(output_dir, exist_ok=True)

        # Process images
        if backend == 'vllm':
            results = wrapper.process_batch_vllm(
                image_paths=image_paths,
                prompt=prompt,
                output_dir=output_dir,
                save_metadata=True
            )
        else:
            results = wrapper.process_batch_transformers(
                image_paths=image_paths,
                prompt=prompt,
                output_dir=output_dir,
                save_metadata=True
            )

        # Update job status
        with job_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['results'] = results
            jobs[job_id]['output_dir'] = output_dir

    except Exception as e:
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['completed_at'] = datetime.now().isoformat()


@app.route('/')
def index():
    """Serve the main web interface"""
    return send_from_directory('.', 'web_interface.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')

        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Get parameters
        prompt = request.form.get('prompt', '<image>\n<|grounding|>Convert the document to markdown.')
        backend = request.form.get('backend', 'vllm')
        mode = request.form.get('mode', 'gundam')

        # Create job ID
        job_id = str(uuid.uuid4())
        upload_dir = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                saved_files.append(filepath)

        if not saved_files:
            return jsonify({'error': 'No valid image files provided'}), 400

        # Create job entry
        with job_lock:
            jobs[job_id] = {
                'id': job_id,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'files': [os.path.basename(f) for f in saved_files],
                'file_count': len(saved_files),
                'prompt': prompt,
                'backend': backend,
                'mode': mode,
                'results': None,
                'error': None
            }

        # Start processing in background thread
        thread = threading.Thread(
            target=process_job,
            args=(job_id, saved_files, prompt, backend, mode)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'file_count': len(saved_files)
        }), 202

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        job_data = jobs[job_id].copy()

        # Don't send full results in status check, just summary
        if job_data['results']:
            job_data['result_count'] = len(job_data['results'])
            job_data['results'] = None

        return jsonify(job_data)


@app.route('/api/job/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get job results"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        job_data = jobs[job_id].copy()

        if job_data['status'] != 'completed':
            return jsonify({'error': 'Job not completed'}), 400

        return jsonify(job_data)


@app.route('/api/job/<job_id>/download/<filename>', methods=['GET'])
def download_result(job_id, filename):
    """Download a result file"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

    output_dir = os.path.join(OUTPUT_FOLDER, job_id)

    if not os.path.exists(os.path.join(output_dir, filename)):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(output_dir, filename)


@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available processing modes"""
    from batch_ocr import OCRModes

    modes = []
    for mode in OCRModes.list_modes():
        modes.append({
            'name': mode.name,
            'description': mode.description,
            'base_size': mode.base_size,
            'image_size': mode.image_size,
            'crop_mode': mode.crop_mode
        })

    return jsonify({'modes': modes})


@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Get preset prompts"""
    prompts = {
        'markdown': {
            'name': 'Convert to Markdown',
            'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
            'description': 'Best for documents, preserves layout and structure'
        },
        'free_ocr': {
            'name': 'Free OCR',
            'prompt': '<image>\nFree OCR.',
            'description': 'Simple text extraction without layout'
        },
        'parse_figure': {
            'name': 'Parse Figure',
            'prompt': '<image>\nParse the figure.',
            'description': 'For charts, graphs, and figures'
        },
        'ocr_image': {
            'name': 'OCR Image',
            'prompt': '<image>\n<|grounding|>OCR this image.',
            'description': 'OCR for general images (not documents)'
        },
        'describe': {
            'name': 'Describe Image',
            'prompt': '<image>\nDescribe this image in detail.',
            'description': 'General image understanding and description'
        }
    }

    return jsonify({'prompts': prompts})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DeepSeek OCR Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'transformers'],
                       help='Default OCR backend')
    parser.add_argument('--mode', default='gundam',
                       choices=['tiny', 'small', 'base', 'large', 'gundam'],
                       help='Default processing mode')

    args = parser.parse_args()

    print("="*60)
    print("DeepSeek OCR Web Server")
    print("="*60)
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Default backend: {args.backend}")
    print(f"Default mode: {args.mode}")
    print("\nOpen http://localhost:5000 in your browser to use the web interface")
    print("="*60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
