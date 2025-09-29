"""
Flask application for ChatGPT-style flooring visualization.

This app serves a front-end interface and provides an API endpoint to
generate room visualizations with a new flooring. When an OpenAI API
key is provided via the ``OPENAI_API_KEY`` environment variable, the
application leverages GPT‑4 Vision to analyze the uploaded room and
flooring images and then uses DALL‑E 3 to generate a realistic
renovation preview. If no API key is configured, the application
falls back to a simple overlay: it replaces the bottom third of the
room image with the selected floor texture. The API returns the
generated image as a data URI so the front end can display it
immediately without needing to fetch external URLs.

The front end is served from ``templates/index.html`` and the
flooring textures reside in ``static/floors``.
"""

import base64
import logging
import os
from io import BytesIO
from typing import Dict, Optional

import requests
try:
    # python-dotenv is optional; if it's not available the application
    # will simply not load environment variables from a .env file.
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None  # type: ignore
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

try:
    from PIL import Image  # type: ignore
except ImportError:
    # Pillow is required for the fallback overlay mode.
    Image = None  # type: ignore

# Load environment variables from a .env file if possible. If the
# python-dotenv package is not installed, this call will be skipped.
if load_dotenv:
    load_dotenv()

app = Flask(
    __name__, static_folder='static', template_folder='templates'
)
# Allow cross‑origin requests to simplify integration in development.
# In production you may want to restrict origins as needed.
CORS(app, origins=["*"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve the OpenAI API key from the environment. When this is
# unavailable, the application will operate in demo mode, simply
# overlaying the selected flooring on the uploaded room image. A
# missing or empty API key will therefore not cause the app to crash.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


@app.route('/')
def index() -> str:
    """Serve the front‑end page."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check() -> "json":
    """Simple health check endpoint for monitoring."""
    return jsonify(
        {
            'message': 'Flooring Visualizer API is running',
            'status': 'healthy',
        }
    )


def get_flooring_name(flooring_id: str) -> str:
    """Map internal flooring identifiers to human‑readable names."""
    flooring_map: Dict[str, str] = {
        'helles-parkett': 'Helles Parkett',
        'dunkles-parkett': 'Dunkles Parkett (Versetzt)',
        'mittleres-parkett': 'Mittleres Parkett (Längs)',
    }
    return flooring_map.get(flooring_id, 'Unbekannter Bodenbelag')


def _prepare_base64(image_str: str) -> str:
    """Strip data URI headers and fix base64 padding.

    Args:
        image_str: A base64 data URI (e.g. 'data:image/jpeg;base64,...') or a
            raw base64 string.

    Returns:
        A raw base64 string with correct padding.
    """
    # Remove the data URI prefix if present
    if image_str.startswith('data:image'):
        image_str = image_str.split(',', 1)[1]
    # Fix padding
    missing_padding = len(image_str) % 4
    if missing_padding:
        image_str += '=' * (4 - missing_padding)
    return image_str


def _overlay_floor(room_b64: str, floor_b64: str) -> str:
    """Overlay the bottom third of the room image with the floor texture.

    This is used when no API key is available. The function decodes the
    provided base64 strings to images, resizes the floor texture to the
    width of the room, crops the floor texture to cover the bottom
    third of the room image, pastes it over the room image, and
    returns the result as a base64 data URI.

    Args:
        room_b64: Base64 encoded room image (without header).
        floor_b64: Base64 encoded floor texture (without header).

    Returns:
        A data URI containing the result image encoded as JPEG.

    Raises:
        RuntimeError: If Pillow is not installed or an error occurs during
            image processing.
    """
    if Image is None:
        raise RuntimeError(
            'Pillow is required for fallback overlay mode but is not installed.'
        )
    try:
        room_img = Image.open(BytesIO(base64.b64decode(room_b64))).convert('RGB')
        floor_img = Image.open(BytesIO(base64.b64decode(floor_b64))).convert('RGB')
    except Exception as exc:
        raise RuntimeError(f'Error decoding images: {exc}') from exc

    width, height = room_img.size
    # Resize the floor texture to the width of the room. Preserve aspect ratio.
    floor_resized = floor_img.resize((width, floor_img.height))
    overlay_height = height // 3
    floor_crop = floor_resized.crop((0, 0, width, overlay_height))

    result_img = room_img.copy()
    result_img.paste(floor_crop, (0, height - overlay_height))

    buffer = BytesIO()
    result_img.save(buffer, format='JPEG')
    out_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{out_b64}'


@app.route('/api/visualize', methods=['POST'])
def visualize_flooring() -> "json":
    """Main API endpoint for generating floor visualizations.

    Expects a JSON payload with the keys ``roomImage``, ``flooringId`` and
    ``flooringImage``. The images should be provided as data URIs or raw
    base64 strings. Returns a JSON response containing either a data URI
    with the generated image (for the fallback mode) or a URL to the
    DALL‑E generated image (when using the OpenAI API), along with
    metadata.
    """
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        logger.error(f"Error parsing JSON body: {exc}")
        return jsonify({'success': False, 'error': 'Invalid JSON'}), 400

    room_image = data.get('roomImage')
    flooring_id = data.get('flooringId')
    flooring_image = data.get('flooringImage')

    # Basic validations
    if not room_image:
        return jsonify({'success': False, 'error': 'Missing roomImage'}), 400
    if not flooring_id:
        return jsonify({'success': False, 'error': 'Missing flooringId'}), 400
    if not flooring_image:
        return jsonify({'success': False, 'error': 'Missing flooringImage'}), 400

    # Prepare base64 data (strip headers and fix padding)
    room_b64 = _prepare_base64(room_image)
    floor_b64 = _prepare_base64(flooring_image)

    # If no API key is configured, return a simple overlay result.
    if not OPENAI_API_KEY:
        try:
            result_data_uri = _overlay_floor(room_b64, floor_b64)
        except Exception as exc:
            logger.error(f"Overlay error: {exc}")
            return jsonify(
                {
                    'success': False,
                    'error': f'Fallback processing failed: {exc}',
                }
            ), 500
        return jsonify(
            {
                'success': True,
                'generatedImageUrl': result_data_uri,
                'method': 'Fallback Overlay',
                'message': 'Demo-Modus: OpenAI API-Schlüssel nicht konfiguriert',
                'flooringName': get_flooring_name(flooring_id),
                'processingTime': 'Demo',
            }
        )

    # Build the prompt for GPT‑4 Vision.
    vision_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Bitte erstelle ein neues Bild von diesem Raum (erstes Bild), "
                            "aber ersetze den Fußboden mit dem Bodenbelag aus dem zweiten Bild. "
                            "Behalte alles andere im Raum exakt gleich - die Möbel, Kartons, Wände, "
                            "Beleuchtung und Perspektive. Nur der Fußboden soll ersetzt werden. "
                            "Das Ergebnis soll wie eine professionelle Renovierungsvisualisierung aussehen."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{room_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{floor_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4000,
    }

    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }

    try:
        logger.info("Using GPT‑4 Vision for analysis...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=vision_data,
            timeout=90,
        )
        if response.status_code != 200:
            logger.error(
                f"Vision API error: {response.status_code} - {response.text[:200]}"
            )
            return jsonify(
                {
                    'success': False,
                    'error': f'Vision analysis failed: {response.text}',
                }
            ), 500
        result = response.json()
        description = result['choices'][0]['message']['content']
        logger.info("GPT‑4 Vision analysis completed.")

        # Use DALL‑E 3 to generate the final image based on the description.
        dalle_data = {
            "model": "dall-e-3",
            "prompt": (
                "Create a realistic interior room visualization based on this description: "
                f"{description}. The image should look like a professional renovation visualization "
                "showing the room with the new flooring installed. High quality, realistic lighting, "
                "professional interior photography."
            ),
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
        }

        dalle_response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=dalle_data,
            timeout=90,
        )
        if dalle_response.status_code != 200:
            logger.error(
                f"DALL‑E error: {dalle_response.status_code} - {dalle_response.text[:200]}"
            )
            return jsonify(
                {
                    'success': False,
                    'error': f'Image generation failed: {dalle_response.text}',
                }
            ), 500

        dalle_result = dalle_response.json()
        generated_image_url = dalle_result['data'][0]['url']
        logger.info("DALL‑E generation successful.")
        return jsonify(
            {
                'success': True,
                'generatedImageUrl': generated_image_url,
                'method': 'ChatGPT-Style: GPT‑4 Vision + DALL‑E 3',
                'message': f'Ihr Raum mit {get_flooring_name(flooring_id)} - ChatGPT-Style Visualisierung',
                'flooringName': get_flooring_name(flooring_id),
                'processingTime': 'Real-time',
                'description': description,
            }
        )

    except Exception as exc:
        logger.exception(f"ChatGPT-style processing error: {exc}")
        return jsonify(
            {
                'success': False,
                'error': f'ChatGPT-style processing failed: {exc}',
            }
        ), 500


if __name__ == '__main__':
    # When running locally, start the development server on port 5000.
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)
