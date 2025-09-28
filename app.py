from flask import Flask, render_template, request, send_file
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')

    """Render the main page of the web application."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Process the uploaded room image and selected floor.
    The selected floor texture is overlaid on the bottom portion of the room image.
    Returns the resulting image directly to the client.
    """
    # Retrieve uploaded room image
    room_file = request.files.get('room_image')
    floor_filename = request.form.get('floor')

    # Validate input
    if not room_file or not floor_filename:
        return ('Missing image or floor selection', 400)

    # Save uploaded image to temporary location
    room_path = os.path.join(app.config['UPLOAD_FOLDER'], 'room_upload.jpg')
    room_file.save(room_path)

    # Determine path for selected floor texture
    floor_dir = os.path.join(app.root_path, 'static
    floor_path = os.path.join(floor_dir, floor_filename)

    # Open images
    room_img = Image.open(room_path).convert('RGB')
    floor_img = Image.open(floor_path).convert('RGB')

    # Resize floor texture to match room width
    floor_resized = floor_img.resize((room_img.width, floor_img.height))

    # Overlay the floor on the bottom one-third of the room image
    overlay_height = room_img.height // 3
    floor_crop = floor_resized.crop((0, 0, room_img.width, overlay_height))
    result_img = room_img.copy()
    result_img.paste(floor_crop, (0, room_img.height - overlay_height))

    # Save result
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    result_img.save(result_path)

    # Send the result image back to the client
    return send_file(result_path, mimetype='image/jpeg')


if __name__ == '__main__':
    # Run the application in debug mode when executed directly
    app.run(host='0.0.0.0', port=5000, debug=True)
