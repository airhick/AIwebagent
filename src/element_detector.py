import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import io
import time

class ElementDetector:
    def __init__(self):
        # Initialize DETR model for visual element detection
        self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        
        # Custom element types for web interface elements
        self.element_types = [
            'button', 'input', 'link', 'text', 'image', 'checkbox',
            'radio', 'dropdown', 'menu', 'icon', 'form', 'header',
            'navigation', 'footer', 'modal', 'popup', 'error', 'loading'
        ]
        
        # Initialize visual memory
        self.visual_memory = []
        self.max_memory_size = 10
        
    def detect_elements(self, image):
        """Detect web elements in image using AI"""
        try:
            # Convert image to proper format
            if isinstance(image, str):
                image = cv2.imread(image)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            
            # Prepare image for model
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Run detection
            outputs = self.model(**inputs)
            
            # Convert outputs to normalized coordinates
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.7
            )[0]
            
            # Process and return detected elements
            elements = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                element_type = self.element_types[int(label) % len(self.element_types)]
                box = [int(i) for i in box.tolist()]
                
                element = {
                    'type': element_type,
                    'bbox': tuple(box),
                    'confidence': float(score),
                    'attributes': self._extract_element_attributes(image, box)
                }
                elements.append(element)
            
            # Update visual memory
            self._update_visual_memory(elements, image)
            
            return elements
            
        except Exception as e:
            print(f"❌ Element detection error: {e}")
            return []
    
    def _extract_element_attributes(self, image, bbox):
        """Extract additional attributes from detected element region"""
        try:
            # Extract region of interest
            x1, y1, x2, y2 = bbox
            element_region = image.crop((x1, y1, x2, y2))
            
            # Convert to numpy array for OpenCV processing
            element_np = np.array(element_region)
            
            attributes = {
                'size': (x2 - x1, y2 - y1),
                'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 > y1 else 0,
                'position': {'x': x1, 'y': y1},
                'is_clickable': self._is_likely_clickable(element_np),
                'has_text': self._has_text(element_np),
                'dominant_color': self._get_dominant_color(element_np),
                'visual_features': self._extract_visual_features(element_np)
            }
            
            return attributes
            
        except Exception as e:
            print(f"⚠️ Attribute extraction error: {e}")
            return {}
    
    def _is_likely_clickable(self, element_np):
        """Determine if element is likely clickable based on visual features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(element_np, cv2.COLOR_RGB2GRAY)
            
            # Look for button-like characteristics
            has_border = cv2.Canny(gray, 100, 200).any()
            has_gradient = np.abs(np.gradient(gray)).mean() > 10
            
            return has_border or has_gradient
            
        except Exception as e:
            print(f"⚠️ Clickable detection error: {e}")
            return False
    
    def _has_text(self, element_np):
        """Detect presence of text in element"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(element_np, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Look for text-like contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours by aspect ratio and size
            text_like = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 0.1 < aspect_ratio < 10 and w*h > 100:
                    text_like += 1
            
            return text_like > 0
            
        except Exception as e:
            print(f"⚠️ Text detection error: {e}")
            return False
    
    def _get_dominant_color(self, element_np):
        """Get dominant color of element"""
        try:
            # Reshape the image
            pixels = element_np.reshape(-1, 3)
            
            # Calculate color clusters
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Get dominant color
            dominant_color = centers[0].astype(np.uint8)
            return tuple(dominant_color)
            
        except Exception as e:
            print(f"⚠️ Color extraction error: {e}")
            return (0, 0, 0)
    
    def _extract_visual_features(self, element_np):
        """Extract visual features for element matching"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(element_np, cv2.COLOR_RGB2GRAY)
            
            # Calculate basic features
            features = {
                'mean_intensity': gray.mean(),
                'std_intensity': gray.std(),
                'edge_density': cv2.Canny(gray, 100, 200).mean(),
                'texture': self._calculate_texture_features(gray)
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return {}
    
    def _calculate_texture_features(self, gray_image):
        """Calculate texture features using GLCM"""
        try:
            # Calculate GLCM matrix
            glcm = np.zeros((256, 256), dtype=np.float32)
            h, w = gray_image.shape
            
            for i in range(h-1):
                for j in range(w-1):
                    i_val = gray_image[i, j]
                    j_val = gray_image[i+1, j+1]
                    glcm[i_val, j_val] += 1
            
            # Normalize GLCM
            glcm = glcm / glcm.sum()
            
            # Calculate texture properties
            contrast = np.sum(np.square(np.arange(256)).reshape(-1, 1) * glcm)
            correlation = np.sum(np.multiply(np.arange(256).reshape(-1, 1), np.arange(256)) * glcm)
            energy = np.sum(np.square(glcm))
            homogeneity = np.sum(glcm / (1 + np.square(np.arange(256)).reshape(-1, 1)))
            
            return {
                'contrast': float(contrast),
                'correlation': float(correlation),
                'energy': float(energy),
                'homogeneity': float(homogeneity)
            }
            
        except Exception as e:
            print(f"⚠️ Texture calculation error: {e}")
            return {}
    
    def _update_visual_memory(self, elements, image):
        """Update visual memory with new elements"""
        try:
            memory_entry = {
                'timestamp': time.time(),
                'elements': elements,
                'image_size': image.size
            }
            
            self.visual_memory.append(memory_entry)
            
            # Keep only recent memories
            if len(self.visual_memory) > self.max_memory_size:
                self.visual_memory.pop(0)
                
        except Exception as e:
            print(f"⚠️ Memory update error: {e}")
    
    def find_similar_elements(self, target_element, threshold=0.8):
        """Find similar elements in visual memory"""
        similar_elements = []
        
        try:
            target_features = target_element.get('attributes', {}).get('visual_features', {})
            if not target_features:
                return similar_elements
            
            for memory in reversed(self.visual_memory):
                for element in memory['elements']:
                    element_features = element.get('attributes', {}).get('visual_features', {})
                    if not element_features:
                        continue
                    
                    similarity = self._calculate_similarity(target_features, element_features)
                    if similarity > threshold:
                        similar_elements.append({
                            'element': element,
                            'similarity': similarity,
                            'timestamp': memory['timestamp']
                        })
            
            return sorted(similar_elements, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            print(f"⚠️ Similarity search error: {e}")
            return similar_elements
    
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        try:
            # Calculate weighted similarity across different feature types
            weights = {
                'mean_intensity': 0.2,
                'std_intensity': 0.2,
                'edge_density': 0.3,
                'texture': 0.3
            }
            
            similarity = 0
            
            # Compare basic features
            for feature in ['mean_intensity', 'std_intensity', 'edge_density']:
                if feature in features1 and feature in features2:
                    diff = abs(features1[feature] - features2[feature])
                    max_val = max(features1[feature], features2[feature])
                    if max_val > 0:
                        similarity += weights[feature] * (1 - diff/max_val)
            
            # Compare texture features
            if 'texture' in features1 and 'texture' in features2:
                texture_sim = self._compare_texture_features(
                    features1['texture'],
                    features2['texture']
                )
                similarity += weights['texture'] * texture_sim
            
            return similarity
            
        except Exception as e:
            print(f"⚠️ Similarity calculation error: {e}")
            return 0
    
    def _compare_texture_features(self, texture1, texture2):
        """Compare texture feature dictionaries"""
        try:
            if not texture1 or not texture2:
                return 0
                
            # Calculate similarity for each texture property
            similarities = []
            for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
                if prop in texture1 and prop in texture2:
                    val1, val2 = texture1[prop], texture2[prop]
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1 - abs(val1 - val2)/max_val
                        similarities.append(similarity)
            
            return sum(similarities)/len(similarities) if similarities else 0
            
        except Exception as e:
            print(f"⚠️ Texture comparison error: {e}")
            return 0 