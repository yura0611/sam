from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from multiprocessing import Process
import base64
from segment_anything import SamPredictor, sam_model_registry
import time
device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
sam.to(device=device)

predictor = SamPredictor(sam)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# server = Process(target=app.run)

current_image = None
current_image_width = 1280
current_image_height = 720
# current_image_width = 2560
# current_image_height = 1440
image_chunks = []
chunk_ranges = []

chunk_size = 512
chunk_stride_x = 256

calculate_chunks = True

sam_features = []


def get_chunk_index(x=1000, y=500):
    global chunk_size
    global chunk_stride_x
    global current_image_width
    stride = chunk_size - chunk_stride_x
    chunks_per_row = (current_image_width + stride - 1) // stride  # Number of chunks per row
    chunks_per_column = (current_image_width + stride - 1) // stride  # Number of chunks per column
    
    # Calculate chunk coordinates
    chunk_x = x // stride
    chunk_y = y // stride
    
    # Handle edge cases where the chunk might be smaller
    if x >= (chunks_per_row - 1) * stride:
        chunk_x = chunks_per_row - 1
    if y >= (chunks_per_column - 1) * stride:
        chunk_y = chunks_per_column - 1
    
    chunk_index = chunk_y * chunks_per_row + chunk_x
    return int(chunk_index)


# def get_chunk_index(x, y):
#     global chunk_size
#     global chunk_overlap
#     chunk_x = x // chunk_size
#     chunk_y = y // chunk_size
#     return int(chunk_y * (x // chunk_size) + chunk_x)

def get_chunk_info(x, y):
    global chunk_size
    global chunk_stride_x
    global current_image_width
    global current_image_height
    global chunk_ranges
    stride = chunk_size - chunk_stride_x
    chunks_per_row = (current_image_width + stride - 1) // stride  # Number of chunks per row
    chunks_per_column = (current_image_height + stride - 1) // stride  # Number of chunks per column
    
    # Calculate chunk coordinates
    chunk_x = x // stride
    chunk_y = y // stride
    
    # Handle edge cases where the chunk might be smaller
    if x >= (chunks_per_row - 1) * stride:
        chunk_x = chunks_per_row - 1
    if y >= (chunks_per_column - 1) * stride:
        chunk_y = chunks_per_column - 1
    
    chunk_index = chunk_y * chunks_per_row + chunk_x
    
    # Get the x and y ranges for the chunk
    x_range, y_range = chunk_ranges[chunk_index]
    local_x = x - x_range[0]
    local_y = y - y_range[0]
    
    return chunk_index, local_x, local_y

def generate_mask(x, y):
    global predictor
    global sam_features
    global current_image_width
    global current_image_height
    global chunk_size
    global image_chunks
    
    stride = chunk_size - chunk_stride_x
    chunk_x = x // stride
    chunk_y = y // stride
    
    chunk_x = x - chunk_x * stride
    chunk_y = y - chunk_y * stride
    
    # chunk_x = x // chunk_size
    # chunk_y = y // chunk_size
    
    
    # chunk_index, chunk_x, chunk_y = get_chunk_info(x, y)
    
    print("Chunk x, y:", chunk_x, chunk_y)
    
    
    # Initialize sam inputs
    input_point = np.array([[chunk_x , chunk_y]])
    input_label = np.array([1])
    
    # Set correct features
    chunk_index = get_chunk_index(x, y)
    print("x, y:", x, y)
    print("Chunk index:", chunk_index)
    chunk = image_chunks[chunk_index]
    # Save chunk locally
    chunk = cv2.cvtColor(chunk, cv2.COLOR_RGB2BGR)
    cv2.imwrite('chunk.png', chunk)

    chunk_height, chunk_width = chunk.shape[:2]
    # Resize chunk to 1024 width while preserving aspect ratio
    # if chunk.shape[1] > 1024:
    #     scale = 1024 / chunk.shape[1]
    #     chunk = cv2.resize(chunk, (1024, int(chunk.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)
    # else:
    #     scale = 1024 / chunk.shape[1]
    #     chunk = cv2.resize(chunk, (1024, int(chunk.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)
        
    predictor.set_image(chunk)
    # predictor.features = sam_features[chunk_index]
    # predictor.is_image_set = True
    # predictor.original_size = (current_image_height, current_image_width)
    # predictor.orig_h = current_image_height
    # predictor.orig_w = current_image_width
    
    # Predict
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Choose mask with highest score
    mask = masks[np.argmax(scores)]
    
    # Post process mask
    # color = np.array([30/255, 144/255, 255/255, 0.6])
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)) * 255.0
    mask_image = mask.reshape(h, w) * 255.0
    
    mask_image = mask_image.astype(np.uint8)
    
    # Set mask as chunk alpha
    chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2BGRA)
    
    chunk[:, :, 3] = mask_image
    
    
    chunk = cv2.resize(chunk, (chunk_width, chunk_height), interpolation=cv2.INTER_LANCZOS4)
    
    
    # Convert mask image to base64
    _, buffer = cv2.imencode('.png', chunk)
    mask_bytes = buffer.tobytes()
    encoded_string = base64.b64encode(mask_bytes).decode('utf-8')
    
    # Save mask image locally
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
    cv2.imwrite('mask.png', mask_image)
    
    return encoded_string

# def chunk_image(image):
#     global chunk_size
#     global chunk_overlap
#     chunks = []
#     h, w = image.shape[:2]
#     stride = chunk_size - chunk_overlap

#     for y in range(0, h, stride):
#         for x in range(0, w, stride):
#             chunk = image[y:y + chunk_size, x:x + chunk_size]
#             if chunk.shape[0] != chunk_size or chunk.shape[1] != chunk_size:
#                 continue
#             chunks.append(chunk)

#     return chunks

def chunk_image(image, chunk_size=512, overlap=256):
    chunks = []
    ranges = []
    h, w = image.shape[:2]
    stride = chunk_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            end_y = min(y + chunk_size, h)
            end_x = min(x + chunk_size, w)
            chunk = image[y:end_y, x:end_x]
            chunks.append(chunk)
            ranges.append(((x, end_x), (y, end_y)))
    
    return chunks, ranges

def calculate_sam_features(chunks):
    global predictor
    global calculate_chunks
    
    if not calculate_chunks:
        # Read all files in sam_features folder
        features = []
        for i in range(len(chunks)):
            features.append(np.load(f'sam_features/feature_{i}.npy'))
        predictor.is_image_set = True
        return features
        
    features = []
    for i, chunk in enumerate(chunks):
        print(chunk.shape)
        # Generate features
        predictor.set_image(chunk)
        
        # Save features
        feature = predictor.features
        feature = feature.detach().cpu().numpy()
        np.save(f'sam_features/feature_{i}.npy', feature)
        
        # Append to list
        features.append(feature)
        
        
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        global current_image
        global image_chunks
        global sam_features
        global chunk_ranges
        
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        current_image = image
        
        # Chunk the image
        image_chunks, chunk_ranges = chunk_image(image)
        print(f'Number of chunks: {len(image_chunks)}')
        
        # Save chunks
        for i, chunk in enumerate(image_chunks):
            cv2.imwrite(f'chunks/chunk_{i}.png', cv2.cvtColor(chunk, cv2.COLOR_RGB2BGR))
        
        # Calculate SAM features
        sam_features = calculate_sam_features(image_chunks)
        
            
        # Encode image as base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'filepath': filepath, 'base64': encoded_string})
    return "File upload failed"

@app.route('/mask', methods=['POST'])
def mask():
    
    data = request.json
    x = data['x']
    y = data['y']
    
    mask = generate_mask(x, y)
    return jsonify({'base64': mask})

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.get('/shutdown')
def shutdown():
    global app
    global server
    server.terminate()
    server.join()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run()
