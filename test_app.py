"""
Unit Tests for Colorimetric Analysis Application

This module contains comprehensive unit tests for all core functions
in the colorimetric analysis application. Tests cover:
    - Image processing and RGB extraction
    - Data saving to CSV
    - Model training
    - Model loading
    - Complete prediction workflow

Test Framework: unittest
Coverage: Core functionality of app.py

Author: Colorimetric Analysis Team
Version: 1.0.0
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestRegressor

# Import functions from app
from app import (
    extract_rgb_from_image,
    save_data_to_csv,
    train_model,
    load_model
)


class TestExtractRGBFromImage(unittest.TestCase):
    """Test cases for extract_rgb_from_image function."""
    
    def create_test_image(self, width, height, color=(100, 150, 200)):
        """
        Helper function to create a test image with uniform color.
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
            color (tuple): RGB color tuple (R, G, B) with values 0-255
        
        Returns:
            BytesIO: File-like object containing PNG image data
        """
        # Create image with specific color (RGB)
        img_array = np.full((height, width, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        # Convert to file-like object
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_buffer.name = 'test_image.png'
        
        return img_buffer
    
    def test_extract_rgb_standard_image(self):
        """Test RGB extraction from standard size image (200x200)."""
        # Create 200x200 image with known color
        test_color = (120, 80, 40)
        img_buffer = self.create_test_image(200, 200, test_color)
        
        # Extract RGB
        r, g, b = extract_rgb_from_image(img_buffer)
        
        # Verify values are close to expected (allowing small tolerance)
        self.assertAlmostEqual(r, test_color[0], delta=1)
        self.assertAlmostEqual(g, test_color[1], delta=1)
        self.assertAlmostEqual(b, test_color[2], delta=1)
    
    def test_extract_rgb_small_image(self):
        """Test RGB extraction from image smaller than ROI (50x50)."""
        # Create 50x50 image (smaller than 100x100 ROI)
        test_color = (200, 100, 50)
        img_buffer = self.create_test_image(50, 50, test_color)
        
        # Extract RGB - should use entire image
        r, g, b = extract_rgb_from_image(img_buffer)
        
        # Verify values match expected color
        self.assertAlmostEqual(r, test_color[0], delta=1)
        self.assertAlmostEqual(g, test_color[1], delta=1)
        self.assertAlmostEqual(b, test_color[2], delta=1)
    
    def test_extract_rgb_large_image(self):
        """Test RGB extraction from large image (500x500)."""
        # Create 500x500 image
        test_color = (150, 200, 100)
        img_buffer = self.create_test_image(500, 500, test_color)
        
        # Extract RGB from center ROI
        r, g, b = extract_rgb_from_image(img_buffer)
        
        # Verify values are close to expected
        self.assertAlmostEqual(r, test_color[0], delta=1)
        self.assertAlmostEqual(g, test_color[1], delta=1)
        self.assertAlmostEqual(b, test_color[2], delta=1)


class TestSaveDataToCSV(unittest.TestCase):
    """Test cases for save_data_to_csv function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, 'test_dataset.csv')
    
    def tearDown(self):
        """Clean up test files."""
        # Remove test CSV if it exists
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        # Remove test directory
        os.rmdir(self.test_dir)
    
    def test_save_data_creates_new_file(self):
        """Test saving data creates new CSV file with header."""
        # Save data to new file
        save_data_to_csv(120.5, 85.3, 45.2, 10.5, self.test_csv)
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.test_csv))
        
        # Verify data was saved correctly
        df = pd.read_csv(self.test_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['R'], 120.5)
        self.assertEqual(df.iloc[0]['G'], 85.3)
        self.assertEqual(df.iloc[0]['B'], 45.2)
        self.assertEqual(df.iloc[0]['Concentration'], 10.5)
    
    def test_save_data_appends_to_existing_file(self):
        """Test saving data appends to existing CSV file."""
        # Save first data point
        save_data_to_csv(100, 80, 60, 5.0, self.test_csv)
        
        # Save second data point
        save_data_to_csv(110, 90, 70, 7.5, self.test_csv)
        
        # Verify both data points exist
        df = pd.read_csv(self.test_csv)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['R'], 110)
        self.assertEqual(df.iloc[1]['Concentration'], 7.5)


class TestTrainModel(unittest.TestCase):
    """Test cases for train_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, 'test_dataset.csv')
        self.test_model = os.path.join(self.test_dir, 'test_model.joblib')
        
        # Create sample dataset
        data = {
            'R': [120, 115, 110, 105, 100, 95],
            'G': [85, 80, 75, 70, 65, 60],
            'B': [45, 42, 40, 38, 35, 32],
            'Concentration': [10.5, 9.8, 9.0, 8.2, 7.5, 6.8]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)
    
    def tearDown(self):
        """Clean up test files."""
        # Remove test files
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.test_model):
            os.remove(self.test_model)
        os.rmdir(self.test_dir)
    
    def test_train_model_success(self):
        """Test successful model training."""
        # Train model
        model, r2_score = train_model(self.test_csv, self.test_model)
        
        # Verify model was created
        self.assertIsNotNone(model)
        self.assertIsInstance(model, RandomForestRegressor)
        
        # Verify R² score is valid
        self.assertIsInstance(r2_score, float)
        self.assertGreaterEqual(r2_score, 0.0)
        self.assertLessEqual(r2_score, 1.0)
        
        # Verify model file was saved
        self.assertTrue(os.path.exists(self.test_model))
    
    def test_train_model_can_predict(self):
        """Test that trained model can make predictions."""
        # Train model
        model, _ = train_model(self.test_csv, self.test_model)
        
        # Test prediction
        test_features = np.array([[115, 80, 42]])
        prediction = model.predict(test_features)
        
        # Verify prediction is reasonable
        self.assertIsInstance(prediction[0], (float, np.floating))
        self.assertGreater(prediction[0], 0)


class TestLoadModel(unittest.TestCase):
    """Test cases for load_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_model = os.path.join(self.test_dir, 'test_model.joblib')
        
        # Create and save a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.array([[100, 80, 60], [110, 90, 70], [120, 100, 80]])
        y = np.array([5.0, 7.5, 10.0])
        model.fit(X, y)
        joblib.dump(model, self.test_model)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_model):
            os.remove(self.test_model)
        os.rmdir(self.test_dir)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        # Load model
        model = load_model(self.test_model)
        
        # Verify model was loaded
        self.assertIsNotNone(model)
        self.assertIsInstance(model, RandomForestRegressor)
        
        # Verify model can predict
        test_features = np.array([[110, 90, 70]])
        prediction = model.predict(test_features)
        self.assertIsInstance(prediction[0], (float, np.floating))


class TestPredictionWorkflow(unittest.TestCase):
    """Test cases for complete prediction workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, 'test_dataset.csv')
        self.test_model = os.path.join(self.test_dir, 'test_model.joblib')
        
        # Create sample dataset
        data = {
            'R': [120, 115, 110, 105, 100],
            'G': [85, 80, 75, 70, 65],
            'B': [45, 42, 40, 38, 35],
            'Concentration': [10.5, 9.8, 9.0, 8.2, 7.5]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.test_model):
            os.remove(self.test_model)
        os.rmdir(self.test_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow: train model, load model, and predict."""
        # Step 1: Train model
        model, r2_score = train_model(self.test_csv, self.test_model)
        self.assertIsNotNone(model)
        
        # Step 2: Load model
        loaded_model = load_model(self.test_model)
        self.assertIsNotNone(loaded_model)
        
        # Step 3: Make prediction
        test_features = np.array([[110, 75, 40]])
        prediction = loaded_model.predict(test_features)
        
        # Verify prediction is reasonable
        self.assertGreater(prediction[0], 0)
        self.assertLess(prediction[0], 20)  # Reasonable range


if __name__ == '__main__':
    unittest.main()
