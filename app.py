import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import os

model_url = 'https://huggingface.co/tristan-gray/acne-detector/resolve/main/best.pt'
model_path = os.path.join("runs/detect/train/weights", "best.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.exists(model_path):
    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Set page config
st.set_page_config(
    page_title="Acne Detection & Skincare",
    page_icon="✨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 1.2em;
    }
    .product-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .header {
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        color: #34495e;
        font-size: 1.8em;
        margin-top: 1.5rem;
    }
    .tip-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .usage-instructions {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'runs/detect/train/weights/best.pt')
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Enhanced skincare recommendations with ingredients, prices, and usage instructions
SKINCARE_RECOMMENDATIONS = {
    'mild': {
        'acne': {
            'cleanser': {
                'name': 'CeraVe Acne Foaming Cream Cleanser',
                'ingredients': 'Benzoyl Peroxide 4%, Ceramides, Hyaluronic Acid, Niacinamide',
                'price': '$14.99',
                'size': '8 oz',
                'usage': [
                    'Wet face with lukewarm water',
                    'Apply a small amount to fingertips',
                    'Gently massage onto face in circular motions',
                    'Rinse thoroughly with water',
                    'Use twice daily, morning and night'
                ]
            },
            'treatment': {
                'name': 'The Ordinary Niacinamide 10% + Zinc 1%',
                'ingredients': 'Niacinamide, Zinc PCA, Hyaluronic Acid',
                'price': '$10.90',
                'size': '30 ml',
                'usage': [
                    'Apply after cleansing and before moisturizing',
                    'Use 2-3 drops for entire face',
                    'Gently pat into skin until absorbed',
                    'Use once or twice daily',
                    'Can be used morning and/or night'
                ]
            },
            'moisturizer': {
                'name': 'La Roche-Posay Effaclar Mat',
                'ingredients': 'Sebulyse Technology, Perlite, Glycerin, Dimethicone',
                'price': '$36.99',
                'size': '1.35 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount for entire face',
                    'Gently massage into skin until absorbed',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Wash your face twice daily with a gentle cleanser',
                'Use non-comedogenic products',
                'Avoid touching your face',
                'Change pillowcases regularly'
            ]
        },
        'blackheads': {
            'cleanser': {
                'name': 'Paula\'s Choice Skin Perfecting 2% BHA Liquid Exfoliant',
                'ingredients': 'Salicylic Acid, Green Tea, Methylpropanediol',
                'price': '$34.00',
                'size': '4 oz',
                'usage': [
                    'Apply to clean, dry skin',
                    'Use a cotton pad to apply over face',
                    'Focus on areas with blackheads',
                    'Use 2-3 times per week initially',
                    'Can increase to daily use if tolerated'
                ]
            },
            'treatment': {
                'name': 'The Ordinary Salicylic Acid 2% Solution',
                'ingredients': 'Salicylic Acid, Hyaluronic Acid, Witch Hazel',
                'price': '$6.80',
                'size': '30 ml',
                'usage': [
                    'Apply after cleansing',
                    'Use 2-3 drops for entire face',
                    'Focus on areas with blackheads',
                    'Allow to absorb before next step',
                    'Use once daily, preferably at night'
                ]
            },
            'moisturizer': {
                'name': 'Neutrogena Hydro Boost Water Gel',
                'ingredients': 'Hyaluronic Acid, Glycerin, Dimethicone',
                'price': '$19.99',
                'size': '1.7 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount',
                    'Gently massage into skin',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Use salicylic acid products',
                'Exfoliate 2-3 times per week',
                'Use oil-free products',
                'Consider professional extraction'
            ]
        }
    },
    'moderate': {
        'acne': {
            'cleanser': {
                'name': 'La Roche-Posay Effaclar Medicated Gel Cleanser',
                'ingredients': '2% Salicylic Acid, Zinc, Glycerin',
                'price': '$15.99',
                'size': '6.8 oz',
                'usage': [
                    'Wet face with lukewarm water',
                    'Apply a small amount to fingertips',
                    'Gently massage onto face in circular motions',
                    'Rinse thoroughly with water',
                    'Use twice daily, morning and night'
                ]
            },
            'treatment': {
                'name': 'Differin Adapalene Gel 0.1%',
                'ingredients': 'Adapalene, Glycerin, Dimethicone',
                'price': '$14.88',
                'size': '15 g',
                'usage': [
                    'Apply a thin layer to clean, dry skin',
                    'Use only at night',
                    'Start with every other night application',
                    'Gradually increase to nightly use',
                    'Always follow with moisturizer'
                ]
            },
            'moisturizer': {
                'name': 'CeraVe PM Facial Moisturizing Lotion',
                'ingredients': 'Ceramides, Niacinamide, Hyaluronic Acid',
                'price': '$16.99',
                'size': '3 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount for entire face',
                    'Gently massage into skin until absorbed',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Use a gentle cleanser with salicylic acid',
                'Apply treatment products consistently',
                'Avoid picking or popping pimples',
                'Consider seeing a dermatologist'
            ]
        },
        'blackheads': {
            'cleanser': {
                'name': 'Neutrogena Oil-Free Acne Wash',
                'ingredients': 'Salicylic Acid 2%, Glycerin, Cocamidopropyl Betaine',
                'price': '$8.99',
                'size': '9.1 oz',
                'usage': [
                    'Wet face with lukewarm water',
                    'Apply a small amount to fingertips',
                    'Gently massage onto face in circular motions',
                    'Rinse thoroughly with water',
                    'Use twice daily, morning and night'
                ]
            },
            'treatment': {
                'name': 'The Inkey List Beta Hydroxy Acid',
                'ingredients': 'Salicylic Acid 2%, Hyaluronic Acid, Aloe Vera',
                'price': '$11.99',
                'size': '30 ml',
                'usage': [
                    'Apply after cleansing',
                    'Use 2-3 drops for entire face',
                    'Focus on areas with blackheads',
                    'Allow to absorb before next step',
                    'Use once daily, preferably at night'
                ]
            },
            'moisturizer': {
                'name': 'Belif The True Cream Aqua Bomb',
                'ingredients': 'Lady\'s Mantle, Oatmeal Extract, Ceramide 3',
                'price': '$38.00',
                'size': '1.68 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount',
                    'Gently massage into skin',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Use a double-cleansing method',
                'Incorporate chemical exfoliants',
                'Use clay masks weekly',
                'Consider professional facials'
            ]
        }
    },
    'severe': {
        'acne': {
            'cleanser': {
                'name': 'PanOxyl Acne Foaming Wash 10% Benzoyl Peroxide',
                'ingredients': 'Benzoyl Peroxide 10%, Glycerin, Stearic Acid',
                'price': '$9.99',
                'size': '5.5 oz',
                'usage': [
                    'Wet face with lukewarm water',
                    'Apply a small amount to fingertips',
                    'Gently massage onto face in circular motions',
                    'Rinse thoroughly with water',
                    'Use twice daily, morning and night'
                ]
            },
            'treatment': {
                'name': 'La Roche-Posay Effaclar Duo Dual Action Acne Treatment',
                'ingredients': 'Benzoyl Peroxide 5.5%, LHA, Niacinamide',
                'price': '$39.99',
                'size': '1.35 oz',
                'usage': [
                    'Apply after cleansing',
                    'Use a pea-sized amount for entire face',
                    'Gently pat into skin until absorbed',
                    'Use once or twice daily',
                    'Always follow with moisturizer'
                ]
            },
            'moisturizer': {
                'name': 'Avene Tolerance Extreme Emulsion',
                'ingredients': 'Squalane, Glycerin, Mineral Oil',
                'price': '$32.00',
                'size': '1.7 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount',
                    'Gently massage into skin',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Consult a dermatologist for prescription treatments',
                'Use gentle, fragrance-free products',
                'Avoid harsh scrubs and exfoliants',
                'Consider medical treatments like Accutane'
            ]
        },
        'blackheads': {
            'cleanser': {
                'name': 'Drunk Elephant Beste No. 9 Jelly Cleanser',
                'ingredients': 'Coconut Surfactants, Glycerin, Marula Oil',
                'price': '$34.00',
                'size': '5 oz',
                'usage': [
                    'Wet face with lukewarm water',
                    'Apply a small amount to fingertips',
                    'Gently massage onto face in circular motions',
                    'Rinse thoroughly with water',
                    'Use twice daily, morning and night'
                ]
            },
            'treatment': {
                'name': 'Sunday Riley UFO Ultra-Clarifying Face Oil',
                'ingredients': 'Salicylic Acid, Tea Tree Oil, Black Cumin Seed Oil',
                'price': '$80.00',
                'size': '1.18 oz',
                'usage': [
                    'Apply after cleansing',
                    'Use 2-3 drops for entire face',
                    'Focus on areas with blackheads',
                    'Gently massage into skin',
                    'Use once daily, preferably at night'
                ]
            },
            'moisturizer': {
                'name': 'First Aid Beauty Ultra Repair Cream',
                'ingredients': 'Colloidal Oatmeal, Shea Butter, Ceramides',
                'price': '$38.00',
                'size': '6 oz',
                'usage': [
                    'Apply after treatment products',
                    'Use a pea-sized amount',
                    'Gently massage into skin',
                    'Use morning and night',
                    'Can be used under makeup'
                ]
            },
            'tips': [
                'Schedule regular professional extractions',
                'Use a combination of BHA and AHA treatments',
                'Consider prescription retinoids',
                'Maintain a consistent skincare routine'
            ]
        }
    }
}

# Add ingredient information dictionary
INGREDIENT_INFO = {
    'Benzoyl Peroxide': {
        'benefits': [
            'Kills acne-causing bacteria',
            'Reduces inflammation',
            'Helps unclog pores',
            'Effective for inflammatory acne'
        ],
        'strength': '2-10%',
        'usage': 'Start with lower concentrations (2-5%) and gradually increase',
        'caution': 'May cause dryness and irritation. Always use sunscreen.'
    },
    'Salicylic Acid': {
        'benefits': [
            'Exfoliates dead skin cells',
            'Unclogs pores',
            'Reduces blackheads and whiteheads',
            'Anti-inflammatory properties'
        ],
        'strength': '0.5-2%',
        'usage': 'Can be used daily or every other day',
        'caution': 'May cause mild irritation. Start with lower frequency.'
    },
    'Niacinamide': {
        'benefits': [
            'Reduces inflammation',
            'Improves skin barrier function',
            'Reduces redness',
            'Helps control oil production'
        ],
        'strength': '2-10%',
        'usage': 'Safe for daily use, morning and night',
        'caution': 'Generally well-tolerated by all skin types'
    },
    'Hyaluronic Acid': {
        'benefits': [
            'Deeply hydrates skin',
            'Plumps and smooths fine lines',
            'Improves skin elasticity',
            'Helps maintain moisture barrier'
        ],
        'strength': '0.1-2%',
        'usage': 'Can be used daily, morning and night',
        'caution': 'Apply to damp skin for best results'
    },
    'Ceramides': {
        'benefits': [
            'Strengthens skin barrier',
            'Locks in moisture',
            'Protects against environmental damage',
            'Reduces sensitivity'
        ],
        'strength': '0.5-5%',
        'usage': 'Best used in moisturizers, morning and night',
        'caution': 'Safe for all skin types'
    },
    'Adapalene': {
        'benefits': [
            'Promotes cell turnover',
            'Unclogs pores',
            'Reduces inflammation',
            'Prevents new breakouts'
        ],
        'strength': '0.1%',
        'usage': 'Start with every other night, can increase to nightly',
        'caution': 'May cause initial dryness and irritation'
    },
    'Tea Tree Oil': {
        'benefits': [
            'Antibacterial properties',
            'Reduces inflammation',
            'Natural alternative to benzoyl peroxide',
            'Helps control oil production'
        ],
        'strength': '5-15%',
        'usage': 'Use 2-3 times per week',
        'caution': 'Always dilute before use. May cause irritation in sensitive skin'
    },
    'Squalane': {
        'benefits': [
            'Deeply moisturizes',
            'Non-comedogenic',
            'Improves skin texture',
            'Suitable for all skin types'
        ],
        'strength': '5-100%',
        'usage': 'Can be used daily, morning and night',
        'caution': 'Generally well-tolerated'
    }
}

class AcneAnalyzer:
    def __init__(self):
        # Initialize classifiers for different characteristics
        self.inflammation_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.severity_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.type_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Train the classifiers with sample data
        self._train_classifiers()
    
    def _train_classifiers(self):
        # Sample training data structure (you would typically load this from a database)
        # Format: [redness, size, confidence, texture_variance, cluster_count, edge_strength]
        X_train = np.array([
            # Low inflammation examples
            [0.2, 0.1, 0.5, 0.3, 2, 0.2],
            [0.3, 0.2, 0.4, 0.2, 1, 0.3],
            # Medium inflammation examples
            [0.5, 0.4, 0.7, 0.5, 3, 0.5],
            [0.6, 0.5, 0.6, 0.6, 4, 0.6],
            # High inflammation examples
            [0.8, 0.7, 0.9, 0.8, 5, 0.8],
            [0.9, 0.8, 0.8, 0.7, 6, 0.9]
        ])
        
        # Labels for inflammation
        y_inflammation = np.array(['low', 'low', 'medium', 'medium', 'high', 'high'])
        # Labels for severity
        y_severity = np.array(['mild', 'mild', 'moderate', 'moderate', 'severe', 'severe'])
        # Labels for type
        y_type = np.array(['blackhead', 'whitehead', 'papule', 'pustule', 'nodule', 'cyst'])
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train the classifiers
        self.inflammation_clf.fit(X_scaled, y_inflammation)
        self.severity_clf.fit(X_scaled, y_severity)
        self.type_clf.fit(X_scaled, y_type)
    
    def extract_features(self, image, detection):
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extract the region of interest (ROI)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Convert to different color spaces
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # 1. Redness analysis (using a* channel from LAB color space)
        redness = np.mean(lab_roi[:, :, 1]) / 255.0
        
        # 2. Size relative to face
        size = ((x2 - x1) * (y2 - y1)) / (image.shape[0] * image.shape[1])
        
        # 3. Model confidence
        confidence = detection['confidence']
        
        # 4. Texture analysis using gray-level co-occurrence matrix (GLCM)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray_roi) / 255.0
        
        # 5. Color clustering to detect different regions
        pixels = roi.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        cluster_count = len(np.unique(kmeans.labels_))
        
        # 6. Edge detection for boundary analysis
        edges = cv2.Canny(gray_roi, 100, 200)
        edge_strength = np.mean(edges) / 255.0
        
        return np.array([redness, size, confidence, texture_variance, cluster_count, edge_strength])
    
    def analyze_acne(self, image, detections):
        if not detections:
            return {
                'severity': 'mild',
                'type': 'unknown',
                'location_zones': set(),
                'inflammation_levels': [],
                'acne_types': [],
                'feature_importances': {}
            }
        
        features_list = []
        valid_detections = []
        
        # Extract features for each detection
        for detection in detections:
            features = self.extract_features(image, detection)
            if features is not None:
                features_list.append(features)
                valid_detections.append(detection)
        
        if not features_list:
            return {
                'severity': 'mild',
                'type': 'unknown',
                'location_zones': set(),
                'inflammation_levels': [],
                'acne_types': [],
                'feature_importances': {}
            }
        
        # Scale features
        X = self.scaler.transform(np.array(features_list))
        
        # Predict characteristics for each detection
        inflammation_levels = self.inflammation_clf.predict(X)
        acne_types = self.type_clf.predict(X)
        
        # Determine overall severity
        severity = self.severity_clf.predict([np.mean(X, axis=0)])[0]
        
        # Calculate location zones
        location_zones = set()
        for det in valid_detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Add zones based on location
            if center_y < image.shape[0] * 0.33:
                location_zones.add('forehead')
            elif center_y > image.shape[0] * 0.66:
                location_zones.add('chin')
            elif center_x < image.shape[1] * 0.4:
                location_zones.add('left_cheek')
            elif center_x > image.shape[1] * 0.6:
                location_zones.add('right_cheek')
            else:
                location_zones.add('t_zone')
        
        # Get feature importances
        feature_names = ['Redness', 'Size', 'Confidence', 'Texture', 'Color Regions', 'Edge Strength']
        importances = {
            'inflammation': dict(zip(feature_names, self.inflammation_clf.feature_importances_)),
            'type': dict(zip(feature_names, self.type_clf.feature_importances_)),
            'severity': dict(zip(feature_names, self.severity_clf.feature_importances_))
        }
        
        return {
            'severity': severity,
            'type': max(set(acne_types), key=list(acne_types).count),  # Most common type
            'location_zones': location_zones,
            'inflammation_levels': list(inflammation_levels),
            'acne_types': list(acne_types),
            'feature_importances': importances
        }

# Initialize the analyzer
acne_analyzer = AcneAnalyzer()

def detect_acne(image):
    # Convert image to RGB if it has 4 channels
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Run inference
    results = model(image)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            detections.append({
                'class': int(cls),
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return detections

def draw_detections(image, detections):
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"Acne: {det['confidence']:.2f}", 
                   (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def select_products(analysis):
    # Initialize scores for each product category
    cleanser_scores = {}
    treatment_scores = {}
    moisturizer_scores = {}
    
    # Map acne types to recommendation categories
    acne_type_mapping = {
        'blackhead': 'blackheads',
        'whitehead': 'acne',
        'papule': 'acne',
        'pustule': 'acne',
        'nodule': 'acne',
        'cyst': 'acne',
        'unknown': 'acne'  # Default case
    }
    
    # Determine predominant inflammation level
    if analysis['inflammation_levels']:
        inflammation_counts = {level: analysis['inflammation_levels'].count(level) 
                            for level in ['low', 'medium', 'high']}
        predominant_inflammation = max(inflammation_counts.items(), key=lambda x: x[1])[0]
    else:
        predominant_inflammation = 'low'
    
    # Map the detected acne type to the recommendation category
    recommendation_type = acne_type_mapping.get(analysis['type'], 'acne')
    
    # Score all products based on characteristics
    for severity in SKINCARE_RECOMMENDATIONS:
        for acne_type in ['acne', 'blackheads']:  # Always check both types
            products = SKINCARE_RECOMMENDATIONS[severity][acne_type]
            
            # Base score multiplier - higher if matches the recommendation type
            type_multiplier = 1.5 if acne_type == recommendation_type else 1.0
            
            # Score cleansers
            cleanser = products['cleanser']
            score = 0
            # Base score from severity match
            if severity == analysis['severity']:
                score += 3
            elif abs(list(SKINCARE_RECOMMENDATIONS.keys()).index(severity) - 
                    list(SKINCARE_RECOMMENDATIONS.keys()).index(analysis['severity'])) == 1:
                score += 1
            
            # Additional scores based on ingredients and characteristics
            if predominant_inflammation == 'high' and 'Benzoyl Peroxide' in cleanser['ingredients']:
                score += 2
            if 't_zone' in analysis['location_zones'] and 'Salicylic Acid' in cleanser['ingredients']:
                score += 1
            
            score *= type_multiplier
            
            cleanser_scores[cleanser['name']] = {
                'product': cleanser, 
                'score': score,
                'reason': f"Selected for {analysis['severity']} {analysis['type']} with {predominant_inflammation} inflammation"
            }
            
            # Score treatments
            treatment = products['treatment']
            score = 0
            if severity == analysis['severity']:
                score += 3
            if predominant_inflammation == 'high' and 'Adapalene' in treatment['ingredients']:
                score += 2
            if len(analysis['location_zones']) > 2 and 'Niacinamide' in treatment['ingredients']:
                score += 1
            
            score *= type_multiplier
            
            treatment_scores[treatment['name']] = {
                'product': treatment, 
                'score': score,
                'reason': f"Targeted treatment for {analysis['type']} acne in {', '.join(analysis['location_zones'])}"
            }
            
            # Score moisturizers
            moisturizer = products['moisturizer']
            score = 0
            if severity == analysis['severity']:
                score += 3
            if predominant_inflammation in ['medium', 'high'] and 'Ceramides' in moisturizer['ingredients']:
                score += 2
            if 'Hyaluronic Acid' in moisturizer['ingredients']:
                score += 1
            
            score *= type_multiplier
            
            moisturizer_scores[moisturizer['name']] = {
                'product': moisturizer, 
                'score': score,
                'reason': f"Hydration suitable for {analysis['severity']} acne-prone skin"
            }
    
    # Select top scoring products
    top_cleanser = max(cleanser_scores.items(), key=lambda x: x[1]['score'])[1]
    top_treatment = max(treatment_scores.items(), key=lambda x: x[1]['score'])[1]
    top_moisturizer = max(moisturizer_scores.items(), key=lambda x: x[1]['score'])[1]
    
    return {
        'cleanser': top_cleanser['product'],
        'cleanser_reason': top_cleanser['reason'],
        'treatment': top_treatment['product'],
        'treatment_reason': top_treatment['reason'],
        'moisturizer': top_moisturizer['product'],
        'moisturizer_reason': top_moisturizer['reason'],
        'tips': SKINCARE_RECOMMENDATIONS[analysis['severity']][recommendation_type]['tips']
    }

def display_product_info(product):
    st.markdown(f"""
        <div class="product-card">
            <h3 style="color: #2c3e50;">{product['name']}</h3>
            <p style="color: #7f8c8d;">Price: {product['price']} ({product['size']})</p>
            <p style="color: #34495e;"><b>Key Ingredients:</b></p>
            <p style="color: #7f8c8d;">{product['ingredients']}</p>
            <p style="color: #34495e; margin-top: 0.5rem;"><b>How to Use:</b></p>
            <div class="usage-instructions">
                <ol style="color: #7f8c8d; margin: 0; padding-left: 1.2rem;">
                    {''.join(f'<li>{step}</li>' for step in product['usage'])}
                </ol>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_ingredient_info(ingredient_name):
    if ingredient_name in INGREDIENT_INFO:
        info = INGREDIENT_INFO[ingredient_name]
        st.markdown(f"""
            <div class="ingredient-card">
                <h3 style="color: #2c3e50;">{ingredient_name}</h3>
                <p style="color: #34495e;"><b>Benefits:</b></p>
                <ul style="color: #7f8c8d;">
                    {''.join(f'<li>{benefit}</li>' for benefit in info['benefits'])}
                </ul>
                <p style="color: #34495e;"><b>Recommended Strength:</b> {info['strength']}</p>
                <p style="color: #34495e;"><b>Usage:</b> {info['usage']}</p>
                <p style="color: #34495e;"><b>Caution:</b> {info['caution']}</p>
            </div>
        """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="header">✨ Acne Detection & Skincare Recommendations ✨</h1>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader_main")
        
        if uploaded_file is not None:
            # Save and process image
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process image
            image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            
            # Detect acne
            detections = detect_acne(image)
            
            # Analyze acne characteristics
            analysis = acne_analyzer.analyze_acne(image, detections)
            
            # Draw detections on image
           # Draw detections on image
            annotated_image = draw_detections(image, detections)

# ❗ OpenCV loads in BGR, Streamlit wants RGB — convert color space
            rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# ✅ Display correctly-colored image
            st.image(rgb_annotated_image, caption="🖼️ Detected Acne (Color-Corrected)", use_column_width=True)
            # Display detection results
            st.markdown("### Detection Results")
            st.write(f"Number of acne spots detected: {len(detections)}")
            
            if len(detections) > 0:
                st.write(f"Severity level: {analysis['severity']}")
                st.write(f"Predominant acne type: {analysis['type']}")
                st.write(f"Affected areas: {', '.join(analysis['location_zones'])}")
                
                # Display inflammation distribution
                st.markdown("### Inflammation Analysis")
                inflammation_counts = {level: analysis['inflammation_levels'].count(level) 
                                    for level in ['low', 'medium', 'high']}
                st.write("Inflammation distribution:")
                for level, count in inflammation_counts.items():
                    st.write(f"- {level.capitalize()}: {count} spots")
                
                # Display feature importance with error handling
                st.markdown("### Feature Importance")
                if 'feature_importances' in analysis and analysis['feature_importances']:
                    st.write("Top factors in determining inflammation:")
                    importances = analysis['feature_importances'].get('inflammation', {})
                    if importances:
                        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        for feature, importance in sorted_features[:3]:
                            st.write(f"- {feature}: {importance:.2f}")
                    else:
                        st.write("No feature importance data available for inflammation.")
                else:
                    st.write("No feature importance data available.")
            else:
                st.success("No acne detected! Your skin looks healthy. 😊")
                st.markdown("""
                    <div class="tip-box">
                        <h3 style="color: #2c3e50;">💡 General Skincare Tips</h3>
                        <ul style="color: #7f8c8d;">
                            <li>Continue with your current skincare routine if it's working well</li>
                            <li>Always wear sunscreen to protect your skin</li>
                            <li>Stay hydrated and maintain a healthy diet</li>
                            <li>Get enough sleep for healthy skin</li>
                            <li>Consider regular skin check-ups</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None and len(detections) > 0:
            # Get personalized recommendations based on the analysis
            recommendations = select_products(analysis)
            
            # Display recommendations
            st.markdown('<h2 class="subheader">Personalized Recommendations</h2>', unsafe_allow_html=True)
            
            # Display product recommendations with explanations
            st.markdown("### Recommended Cleanser")
            display_product_info(recommendations['cleanser'])
            st.write("Selected based on:", recommendations['cleanser_reason'])
            
            st.markdown("### Recommended Treatment")
            display_product_info(recommendations['treatment'])
            st.write("Selected based on:", recommendations['treatment_reason'])
            
            st.markdown("### Recommended Moisturizer")
            display_product_info(recommendations['moisturizer'])
            st.write("Selected based on:", recommendations['moisturizer_reason'])
            
            # Display tips
            st.markdown("""
                <div class="tip-box">
                    <h3 style="color: #2c3e50;">💡 Personalized Skincare Tips</h3>
                    <ul style="color: #7f8c8d;">
            """, unsafe_allow_html=True)
            
            for tip in recommendations['tips']:
                st.markdown(f'<li>{tip}</li>', unsafe_allow_html=True)
            
            st.markdown('</ul></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


