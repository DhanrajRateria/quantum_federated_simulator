"""
Communication Module for Quantum Federated Learning Simulator

This module implements communication protocols between nodes in the federated
network, optimizing bandwidth usage through efficient serialization and
compression for quantum model parameters.
"""

import logging
import pickle
import gzip
import time
import socket
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np

# Import compression libraries
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

class MessageType:
    """Enumeration of message types in the communication protocol."""
    MODEL_UPDATE = "model_update"
    PARAMETER_REQUEST = "parameter_request"
    GLOBAL_MODEL = "global_model"
    HEARTBEAT = "heartbeat"
    NODE_STATUS = "node_status"
    COMMAND = "command"
    METRICS = "metrics"
    ERROR = "error"

class CompressionType:
    """Enumeration of compression algorithms available."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

class CommunicationManager:
    """
    Manages communication protocols between nodes in the federated learning network.
    Handles parameter serialization, compression, and network communication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Communication Manager with configuration parameters.
        
        Args:
            config: Dictionary containing communication configuration
        """
        self.config = config
        self.compression = config.get("compression", CompressionType.GZIP)
        self.compression_level = config.get("compression_level", 1)
        self.message_timeout = config.get("message_timeout", 60)  # seconds
        self.max_retries = config.get("max_retries", 3)
        self.buffer_size = config.get("buffer_size", 4096)
        self.logger = logging.getLogger(__name__)
        
        # Message handlers
        self.message_handlers = {}
        
        # Statistics for monitoring
        self.stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "compression_ratio": 0,
            "failed_messages": 0
        }
        
        # Check if compression types are available
        if self.compression == CompressionType.LZ4 and not HAS_LZ4:
            self.logger.warning("LZ4 compression requested but not available. Falling back to GZIP.")
            self.compression = CompressionType.GZIP
            
        if self.compression == CompressionType.ZSTD and not HAS_ZSTD:
            self.logger.warning("Zstandard compression requested but not available. Falling back to GZIP.")
            self.compression = CompressionType.GZIP
    
    def register_handler(self, message_type: str, handler_func: Callable) -> None:
        """
        Register a handler function for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler_func: Function to call when message is received
        """
        self.message_handlers[message_type] = handler_func
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    def serialize_parameters(self, parameters: Dict[str, Any]) -> bytes:
        """
        Serialize model parameters with optimization for quantum parameters.
        
        Args:
            parameters: Dictionary of model parameters
            
        Returns:
            Serialized binary data
        """
        try:
            # Basic serialization
            data = pickle.dumps(parameters)
            
            # Apply compression if enabled
            original_size = len(data)
            compressed_data = self._compress_data(data)
            compressed_size = len(compressed_data)
            
            # Update compression ratio statistic
            if original_size > 0:
                self.stats["compression_ratio"] = compressed_size / original_size
                
            self.logger.debug(f"Serialized {original_size} bytes to {compressed_size} bytes "
                            f"(ratio: {self.stats['compression_ratio']:.2f})")
            
            return compressed_data
        except Exception as e:
            self.logger.error(f"Serialization error: {str(e)}")
            raise
    
    def deserialize_parameters(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize model parameters.
        
        Args:
            data: Serialized binary data
            
        Returns:
            Dictionary of model parameters
        """
        try:
            # Decompress
            decompressed_data = self._decompress_data(data)
            
            # Deserialize
            parameters = pickle.loads(decompressed_data)
            
            return parameters
        except Exception as e:
            self.logger.error(f"Deserialization error: {str(e)}")
            raise
    
    def _compress_data(self, data: bytes) -> bytes:
        """
        Compress binary data using selected algorithm.
        
        Args:
            data: Raw binary data
            
        Returns:
            Compressed binary data
        """
        if self.compression == CompressionType.NONE:
            return data
        
        elif self.compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=self.compression_level)
        
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            return lz4.frame.compress(data, compression_level=self.compression_level)
        
        elif self.compression == CompressionType.ZSTD and HAS_ZSTD:
            compressor = zstandard.ZstdCompressor(level=self.compression_level)
            return compressor.compress(data)
        
        else:
            # Default to GZIP
            return gzip.compress(data, compresslevel=self.compression_level)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """
        Decompress binary data.
        
        Args:
            data: Compressed binary data
            
        Returns:
            Decompressed binary data
        """
        if self.compression == CompressionType.NONE:
            return data
        
        elif self.compression == CompressionType.GZIP:
            return gzip.decompress(data)
        
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            return lz4.frame.decompress(data)
        
        elif self.compression == CompressionType.ZSTD and HAS_ZSTD:
            decompressor = zstandard.ZstdDecompressor()
            return decompressor.decompress(data)
        
        else:
            # Default to GZIP
            return gzip.decompress(data)
    
    def send_message(self, destination: str, message_type: str, 
                    payload: Dict[str, Any]) -> bool:
        """
        Send a message to another node.
        
        Args:
            destination: Destination address (host:port)
            message_type: Type of message
            payload: Message content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message structure
            message = {
                "type": message_type,
                "sender": self.config.get("node_id", "unknown"),
                "timestamp": time.time(),
                "payload": payload
            }
            
            # Serialize and compress
            data = self.serialize_parameters(message)
            
            # Send message
            host, port = destination.split(":")
            port = int(port)
            
            for attempt in range(self.max_retries):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(self.message_timeout)
                        sock.connect((host, port))
                        
                        # Send message length first
                        length_bytes = len(data).to_bytes(8, byteorder='big')
                        sock.sendall(length_bytes)
                        
                        # Send message data
                        sock.sendall(data)
                        
                        # Update statistics
                        self.stats["bytes_sent"] += len(data)
                        self.stats["messages_sent"] += 1
                        
                        self.logger.debug(f"Sent {message_type} message to {destination} "
                                         f"({len(data)} bytes)")
                        return True
                except (socket.timeout, ConnectionRefusedError, OSError) as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Retrying message to {destination} "
                                           f"(attempt {attempt+1}/{self.max_retries})")
                        time.sleep(1)  # Wait before retry
                    else:
                        self.logger.error(f"Failed to send message to {destination}: {str(e)}")
                        self.stats["failed_messages"] += 1
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            self.stats["failed_messages"] += 1
            return False
    
    def start_receiver(self, host: str, port: int) -> threading.Thread:
        """
        Start a receiver thread to listen for incoming messages.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            
        Returns:
            The started receiver thread
        """
        receiver_thread = threading.Thread(
            target=self._receiver_loop,
            args=(host, port),
            daemon=True
        )
        receiver_thread.start()
        return receiver_thread
    
    def _receiver_loop(self, host: str, port: int) -> None:
        """
        Main loop for the receiver thread.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
        """
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(10)
            
            self.logger.info(f"Communication receiver started on {host}:{port}")
            
            while True:
                client_socket, client_address = server_socket.accept()
                client_handler = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_handler.start()
                
        except Exception as e:
            self.logger.error(f"Receiver error: {str(e)}")
        finally:
            server_socket.close()
    
    def _handle_client(self, client_socket: socket.socket, 
                      client_address: Tuple[str, int]) -> None:
        """
        Handle a client connection.
        
        Args:
            client_socket: Socket for client communication
            client_address: Client address (host, port)
        """
        try:
            client_socket.settimeout(self.message_timeout)
            
            # Read message length
            length_bytes = client_socket.recv(8)
            if not length_bytes:
                return
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Read message data
            chunks = []
            bytes_received = 0
            
            while bytes_received < message_length:
                chunk = client_socket.recv(min(self.buffer_size, message_length - bytes_received))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)
            
            if bytes_received < message_length:
                self.logger.warning(f"Incomplete message received from {client_address}")
                return
            
            # Combine chunks and deserialize
            data = b''.join(chunks)
            message = self.deserialize_parameters(data)
            
            # Update statistics
            self.stats["bytes_received"] += len(data)
            self.stats["messages_received"] += 1
            
            # Process message
            message_type = message.get("type")
            sender = message.get("sender", "unknown")
            
            self.logger.debug(f"Received {message_type} message from {sender} "
                             f"({len(data)} bytes)")
            
            # Handle message based on type
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                handler(message.get("payload", {}), sender)
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
            
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {str(e)}")
        finally:
            client_socket.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get communication statistics.
        
        Returns:
            Dictionary of communication statistics
        """
        return self.stats.copy()
    
    def optimize_parameters_for_transmission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters before transmission to reduce size.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Optimized parameters
        """
        optimized = {}
        
        for key, value in parameters.items():
            # Handle numpy arrays efficiently
            if isinstance(value, np.ndarray):
                # Convert to float16 for transmission if it's a floating point array
                if np.issubdtype(value.dtype, np.floating):
                    optimized[key] = value.astype(np.float16)
                else:
                    optimized[key] = value
            else:
                optimized[key] = value
        
        return optimized
    
    def restore_parameters_after_transmission(self, 
                                             parameters: Dict[str, Any], 
                                             original_dtypes: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Restore parameters to original format after transmission.
        
        Args:
            parameters: Received parameters
            original_dtypes: Original data types (if known)
            
        Returns:
            Restored parameters
        """
        restored = {}
        
        for key, value in parameters.items():
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                if original_dtypes and key in original_dtypes:
                    # Restore original dtype if provided
                    restored[key] = value.astype(original_dtypes[key])
                elif np.issubdtype(value.dtype, np.floating) and value.dtype == np.float16:
                    # Restore to float32 by default
                    restored[key] = value.astype(np.float32)
                else:
                    restored[key] = value
            else:
                restored[key] = value
        
        return restored