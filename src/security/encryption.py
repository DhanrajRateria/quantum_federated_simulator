"""
Encryption mechanisms for secure parameter exchange in quantum federated learning.
This module provides tools to encrypt, decrypt, and protect model parameters
during transmission between clients and server.
"""

import numpy as np
import base64
import json
import os
from typing import Dict, Tuple, Union, Optional
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class ParameterEncryption:
    """Base class for parameter encryption methods."""
    
    def encrypt(self, parameters: np.ndarray) -> Dict:
        """
        Encrypt model parameters.
        
        Args:
            parameters: Model parameters to encrypt
            
        Returns:
            Dictionary containing encrypted parameters
        """
        raise NotImplementedError
    
    def decrypt(self, encrypted_data: Dict) -> np.ndarray:
        """
        Decrypt model parameters.
        
        Args:
            encrypted_data: Dictionary containing encrypted parameters
            
        Returns:
            Decrypted model parameters
        """
        raise NotImplementedError

class AESEncryption(ParameterEncryption):
    """AES encryption for model parameters."""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize with AES key.
        
        Args:
            key: AES key (generated if None)
        """
        if key is None:
            # Generate a random 256-bit key
            self.key = os.urandom(32)
        else:
            self.key = key
    
    def encrypt(self, parameters: np.ndarray) -> Dict:
        """
        Encrypt model parameters using AES.
        
        Args:
            parameters: Model parameters to encrypt
            
        Returns:
            Dictionary with encrypted parameters
        """
        # Serialize parameters
        serialized = json.dumps({
            'shape': parameters.shape,
            'dtype': str(parameters.dtype),
            'data': parameters.tolist()
        }).encode('utf-8')
        
        # Generate random IV
        iv = os.urandom(16)
        
        # Create AES cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CFB(iv),
            backend=default_backend()
        )
        
        # Encrypt
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(serialized) + encryptor.finalize()
        
        # Return as base64 strings for safe transmission
        return {
            'iv': base64.b64encode(iv).decode('utf-8'),
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8')
        }
    
    def decrypt(self, encrypted_data: Dict) -> np.ndarray:
        """
        Decrypt model parameters.
        
        Args:
            encrypted_data: Dictionary with encrypted parameters
            
        Returns:
            Decrypted model parameters
        """
        # Decode from base64
        iv = base64.b64decode(encrypted_data['iv'])
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        
        # Create AES cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CFB(iv),
            backend=default_backend()
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Deserialize
        json_data = json.loads(plaintext.decode('utf-8'))
        
        # Convert back to numpy array
        shape = tuple(json_data['shape'])
        dtype = np.dtype(json_data['dtype'])
        return np.array(json_data['data'], dtype=dtype).reshape(shape)
    
    def export_key(self) -> str:
        """
        Export encryption key as base64 string.
        
        Returns:
            Base64 encoded key
        """
        return base64.b64encode(self.key).decode('utf-8')
    
    @classmethod
    def from_key(cls, key_base64: str) -> 'AESEncryption':
        """
        Create encryption object from base64 key.
        
        Args:
            key_base64: Base64 encoded key
            
        Returns:
            Initialized AESEncryption instance
        """
        key = base64.b64decode(key_base64)
        return cls(key)

class RSAEncryption(ParameterEncryption):
    """RSA encryption for model parameters."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize with RSA key pair.
        
        Args:
            key_size: Size of RSA key in bits
        """
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def get_public_key_pem(self) -> str:
        """
        Get PEM-encoded public key.
        
        Returns:
            PEM-encoded public key
        """
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    def encrypt(self, parameters: np.ndarray) -> Dict:
        """
        Encrypt model parameters using RSA + AES.
        For large parameters, use hybrid encryption (RSA for AES key, AES for data).
        
        Args:
            parameters: Model parameters to encrypt
            
        Returns:
            Dictionary with encrypted parameters
        """
        # Use AES for data encryption (RSA has size limitations)
        aes = AESEncryption()
        aes_encrypted = aes.encrypt(parameters)
        
        # Encrypt AES key with RSA
        encrypted_key = self.public_key.encrypt(
            base64.b64decode(aes.export_key()),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'iv': aes_encrypted['iv'],
            'ciphertext': aes_encrypted['ciphertext']
        }
    
    def decrypt(self, encrypted_data: Dict) -> np.ndarray:
        """
        Decrypt model parameters.
        
        Args:
            encrypted_data: Dictionary with encrypted parameters
            
        Returns:
            Decrypted model parameters
        """
        # Decrypt AES key with RSA
        encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
        aes_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Use AES for data decryption
        aes = AESEncryption(aes_key)
        return aes.decrypt({
            'iv': encrypted_data['iv'],
            'ciphertext': encrypted_data['ciphertext']
        })

class HomomorphicEncryptionSimulator(ParameterEncryption):
    """
    Simulated homomorphic encryption for educational purposes.
    
    Note: This is NOT real homomorphic encryption, just a simulation for educational
    purposes. It provides the same API but only adds obscurity, not security.
    For real applications, use a proper homomorphic encryption library.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize with random seed.
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        self.key = np.random.rand(10)
    
    def encrypt(self, parameters: np.ndarray) -> Dict:
        """
        "Encrypt" parameters in a way that still allows certain operations.
        
        Args:
            parameters: Model parameters to "encrypt"
            
        Returns:
            Dictionary with "encrypted" parameters
        """
        # Apply simple transformations that maintain addition/multiplication properties
        # This is NOT real homomorphic encryption
        
        # Generate random "mask" with same shape as parameters
        mask = np.random.rand(*parameters.shape)
        scale = np.random.rand(*parameters.shape) + 0.5  # Prevent zero scales
        
        # Apply affine transformation
        transformed = scale * parameters + mask
        
        return {
            'data': transformed.tolist(),
            'shape': parameters.shape,
            'dtype': str(parameters.dtype),
            'mask': mask.tolist(),
            'scale': scale.tolist()
        }
    
    def decrypt(self, encrypted_data: Dict) -> np.ndarray:
        """
        "Decrypt" parameters.
        
        Args:
            encrypted_data: Dictionary with "encrypted" parameters
            
        Returns:
            Decrypted model parameters
        """
        # Convert to numpy arrays
        shape = tuple(encrypted_data['shape'])
        dtype = np.dtype(encrypted_data['dtype'])
        
        transformed = np.array(encrypted_data['data'], dtype=dtype).reshape(shape)
        mask = np.array(encrypted_data['mask'], dtype=dtype).reshape(shape)
        scale = np.array(encrypted_data['scale'], dtype=dtype).reshape(shape)
        
        # Reverse affine transformation
        original = (transformed - mask) / scale
        
        return original
    
    def add_encrypted(self, encrypted_a: Dict, encrypted_b: Dict) -> Dict:
        """
        Add two "encrypted" parameter sets.
        
        Args:
            encrypted_a: First encrypted parameters
            encrypted_b: Second encrypted parameters
            
        Returns:
            Sum of encrypted parameters
        """
        # Verify shapes match
        if encrypted_a['shape'] != encrypted_b['shape']:
            raise ValueError("Shape mismatch in encrypted addition")
        
        shape = tuple(encrypted_a['shape'])
        dtype = np.dtype(encrypted_a['dtype'])
        
        # Convert to numpy arrays
        data_a = np.array(encrypted_a['data'], dtype=dtype).reshape(shape)
        data_b = np.array(encrypted_b['data'], dtype=dtype).reshape(shape)
        
        # Add encrypted values
        result_data = data_a + data_b
        
        # Need to adjust mask and scale for correct decryption
        mask_a = np.array(encrypted_a['mask'], dtype=dtype).reshape(shape)
        mask_b = np.array(encrypted_b['mask'], dtype=dtype).reshape(shape)
        scale_a = np.array(encrypted_a['scale'], dtype=dtype).reshape(shape)
        scale_b = np.array(encrypted_b['scale'], dtype=dtype).reshape(shape)
        
        # Combined transformation parameters
        result_mask = mask_a + mask_b
        result_scale = (scale_a + scale_b) / 2
        
        return {
            'data': result_data.tolist(),
            'shape': encrypted_a['shape'],
            'dtype': encrypted_a['dtype'],
            'mask': result_mask.tolist(),
            'scale': result_scale.tolist()
        }