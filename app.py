from flask import Flask, request, jsonify, render_template
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
 
# Start Flask app (the website)
app = Flask(__name__)
 
# Allow 500MB file uploads
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
 
# Device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Load the classification model
def load_model():
    MODEL_PATH = "resnet18_restained_model_best.pth"
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return None, None
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["classes"]
    num_classes = len(class_names)
    
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded: {num_classes} classes → {class_names}")
    return model, class_names
 
# Load model at startup
model, class_names = load_model()
 
# Function to predict
def predict_image(img_path):
    try:
        infer_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(img_path).convert("RGB")
        tensor_img = infer_transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            import torch.nn.functional as F
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        
        predicted_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item() * 100
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
 
# ROUTE 1: Home page (user sees upload form)
@app.route('/', methods=['GET'])
def index():
    """Show the upload website"""
    return render_template('index.html')
 
# ROUTE 2: Upload endpoint (user sends image here)
@app.route('/api/upload', methods=['POST'])
def upload_images():
    """
    User uploads images
    We classify them
    We return results
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    predictions = []
    
    for f in files:
        if f.filename:
            # Save temporarily
            temp_path = f"/tmp/{f.filename}"
            f.save(temp_path)
            
            # Predict
            result = predict_image(temp_path)
            
            if result['success']:
                predictions.append({
                    'filename': f.filename,
                    'predicted_class': result['predicted_class'],
                    'confidence': round(result['confidence'], 2)
                })
            
            # Delete temp file
            try:
                os.remove(temp_path)
            except:
                pass
    
    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'total': len(predictions)
    }), 200
 
# ROUTE 3: Health check (Render pings this to see if alive)
@app.route('/api/health', methods=['GET'])
def health():
    """Check if server is alive"""
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'model_loaded': model is not None
    }), 200
 
# ROUTE 4: Get available classes
@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Tell user what tumor types we can detect"""
    if not class_names:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': class_names,
        'num_classes': len(class_names)
    }), 200
 
# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 500MB)'}), 413
 
@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': str(e)}), 500
 
# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)