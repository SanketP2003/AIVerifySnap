"""
AIVerifySnap ML Service Package.

A production-ready deepfake detection microservice using a dual-stream
hybrid neural network (AIVerifyNet) combining:
- Spatial analysis (RGB features via ResNet-50)
- Frequency analysis (ELA features via ResNet-18)

Modules:
- main: FastAPI application and endpoints
- model: Detection model wrapper
- aiverifynet: Dual-stream neural network architecture
- ela_utils: Error Level Analysis utilities
- preprocess: Image preprocessing functions
- schemas: Pydantic request/response models
- config: Application configuration
"""

__version__ = "1.0.0"
__author__ = "AIVerifySnap Team"

