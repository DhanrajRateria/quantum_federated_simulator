"""
Communication module for the Quantum Federated Learning Simulator.

This module handles the communication between nodes in the federated learning
system, including serialization/deserialization, encryption, compression,
and network protocols.
"""

import logging
import pickle
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq
from cryptography.fernet import Fernet


class CommunicationProtocol(Enum):
    """Communication protocols supported by the system."""
    TCP = "tcp"
    UDP = "udp"
    ZMQ = "zmq"
    GRPC = "grpc"
    HTTP = "http"


class Communication:
    """
    Handles communication between nodes in the federated learning system.
    
    This class provides methods for sending and receiving data between nodes,
    with support for different protocols, compression, and encryption.
    """
    
    def __init__(self, protocol: CommunicationProtocol = CommunicationProtocol.ZMQ,
                 encryption_key: Optional[str] = None, 
                 compression_level: int = 0,
                 timeout: int = 30000):
        """
        Initialize the Communication instance.
        
        Args:
            protocol: Communication protocol to use
            encryption_key: Key for encrypting communications (None for no encryption)
            compression_level: Level of compression (0-9, 0 is no compression)
            timeout: Communication timeout in milliseconds
        """
        self.logger = logging.getLogger(__name__)
        self.protocol = protocol
        self.compression_level = compression_level
        self.timeout = timeout
        
        # Setup encryption if key provided
        self.encryption = None
        if encryption_key:
            if isinstance(encryption_key, str):
                # If string is provided, convert to bytes
                encryption_key = encryption_key.encode('utf-8')
            self.encryption = Fernet(encryption_key)
        
        # Initialize context for ZMQ
        if protocol == CommunicationProtocol.ZMQ:
            self.context = zmq.Context()
            
        self.sockets = {}
        self.logger.info(f"Communication initialized with protocol: {protocol.value}")
    
    def create_socket(self, socket_type: str, identity: str = None) -> Any:
        """
        Create a socket of the specified type.
        
        Args:
            socket_type: Type of socket to create (e.g., 'PUSH', 'PULL')
            identity: Identity for the socket (for ZMQ)
            
        Returns:
            The created socket
        """
        if self.protocol == CommunicationProtocol.ZMQ:
            socket_type_enum = getattr(zmq, socket_type)
            socket = self.context.socket(socket_type_enum)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            
            if identity:
                socket.setsockopt(zmq.IDENTITY, identity.encode('utf-8'))
            
            return socket
        
        # For other protocols, implement similar socket creation logic
        raise NotImplementedError(f"Socket creation for {self.protocol.value} not implemented yet")
    
    def bind(self, socket: Any, address: str) -> None:
        """
        Bind a socket to an address.
        
        Args:
            socket: Socket to bind
            address: Address to bind to (e.g., 'tcp://*:5555')
        """
        full_address = f"{self.protocol.value}://{address}"
        socket.bind(full_address)
        self.sockets[address] = socket
        self.logger.debug(f"Socket bound to {full_address}")
    
    def connect(self, socket: Any, address: str) -> None:
        """
        Connect a socket to an address.
        
        Args:
            socket: Socket to connect
            address: Address to connect to (e.g., 'localhost:5555')
        """
        full_address = f"{self.protocol.value}://{address}"
        socket.connect(full_address)
        self.sockets[address] = socket
        self.logger.debug(f"Socket connected to {full_address}")
    
    def serialize(self, data: Any) -> bytes:
        """
        Serialize data for transmission.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized bytes
        """
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress if configured
        if self.compression_level > 0:
            import zlib
            serialized = zlib.compress(serialized, level=self.compression_level)
            
        # Encrypt if configured
        if self.encryption:
            serialized = self.encryption.encrypt(serialized)
            
        return serialized
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize received data.
        
        Args:
            data: Serialized data bytes
            
        Returns:
            Deserialized data
        """
        # Decrypt if configured
        if self.encryption:
            data = self.encryption.decrypt(data)
            
        # Decompress if configured
        if self.compression_level > 0:
            import zlib
            data = zlib.decompress(data)
            
        return pickle.loads(data)
    
    def send(self, socket: Any, data: Any, flags: int = 0) -> bool:
        """
        Send data over a socket.
        
        Args:
            socket: Socket to send data through
            data: Data to send
            flags: Additional flags for the send operation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = self.serialize(data)
            if self.protocol == CommunicationProtocol.ZMQ:
                socket.send(serialized, flags=flags)
            else:
                # Implement for other protocols
                pass
            return True
        except Exception as e:
            self.logger.error(f"Error sending data: {e}")
            return False
    
    def receive(self, socket: Any, flags: int = 0) -> Tuple[bool, Any]:
        """
        Receive data from a socket.
        
        Args:
            socket: Socket to receive data from
            flags: Additional flags for the receive operation
            
        Returns:
            Tuple of (success, data)
        """
        try:
            if self.protocol == CommunicationProtocol.ZMQ:
                message = socket.recv(flags=flags)
                data = self.deserialize(message)
                return True, data
            else:
                # Implement for other protocols
                pass
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                self.logger.debug("No message received within timeout")
            else:
                self.logger.error(f"ZMQ error receiving data: {e}")
            return False, None
        except Exception as e:
            self.logger.error(f"Error receiving data: {e}")
            return False, None
    
    def close(self) -> None:
        """Close all sockets and cleanup resources."""
        for addr, socket in self.sockets.items():
            socket.close()
            self.logger.debug(f"Socket for {addr} closed")
        
        self.sockets.clear()
        
        if self.protocol == CommunicationProtocol.ZMQ:
            self.context.term()
            
        self.logger.info("Communication resources released")