

import os
import io
import zlib
import numpy as np
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes



def _load_key() -> bytes:
    """
    Load the 32-byte shared key from the FL_SHARED_KEY environment variable.
    The variable must be a 64-character hex string (32 bytes).

    Raises RuntimeError if the variable is missing or the wrong length.
    """
    hex_key = os.environ.get("FL_SHARED_KEY", "")
    if not hex_key:
        raise RuntimeError(
            "FL_SHARED_KEY environment variable is not set.\n"
            "Generate one with:  python -c \"import os; print(os.urandom(32).hex())\"\n"
            "Then set it before running server or client."
        )
    try:
        key = bytes.fromhex(hex_key)
    except ValueError:
        raise RuntimeError("FL_SHARED_KEY must be a valid hex string.")
    if len(key) != 32:
        raise RuntimeError(
            f"FL_SHARED_KEY must decode to exactly 32 bytes, got {len(key)}."
        )
    return key


SHARED_KEY: bytes = _load_key()

class ChaCha20Cipher:

    def __init__(self, key: bytes = SHARED_KEY):
        if len(key) != 32:
            raise ValueError("ChaCha20-Poly1305 requires exactly a 32-byte key.")
        self.key = key

    def encrypt(self, data: bytes) -> bytes:
        """Returns nonce (12 B) + tag (16 B) + ciphertext."""
        nonce = get_random_bytes(12)
        cipher = ChaCha20_Poly1305.new(key=self.key, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        return nonce + tag + ciphertext

    def decrypt(self, payload: bytes) -> bytes:
        """
        Decrypts and verifies integrity.
        Raises ValueError if the payload has been tampered with.
        """
        if len(payload) < 28:  # 12 nonce + 16 tag minimum
            raise ValueError("Payload too short to be a valid ChaCha20-Poly1305 ciphertext.")
        nonce      = payload[:12]
        tag        = payload[12:28]
        ciphertext = payload[28:]
        cipher = ChaCha20_Poly1305.new(key=self.key, nonce=nonce)
        # verify() raises ValueError on MAC mismatch — stops malicious payloads
        return cipher.decrypt_and_verify(ciphertext, tag)

def _serialize_arrays(arrays: list) -> bytes:
    """Serialize a list of np.ndarray to bytes using numpy's .npy format (no pickle)."""
    buf = io.BytesIO()
    # Write count header (4 bytes, big-endian uint32)
    buf.write(len(arrays).to_bytes(4, "big"))
    for arr in arrays:
        np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _deserialize_arrays(data: bytes) -> list:
    """Deserialize bytes produced by _serialize_arrays back to a list of np.ndarray."""
    buf = io.BytesIO(data)
    count = int.from_bytes(buf.read(4), "big")
    arrays = []
    for _ in range(count):
        arrays.append(np.load(buf, allow_pickle=False))
    return arrays

def pack_parameters(params: list, cipher: ChaCha20Cipher) -> np.ndarray:
    """Bundle, compress, and encrypt a list of numpy arrays into a single uint8 array."""
    raw        = _serialize_arrays(params)
    compressed = zlib.compress(raw, level=6)
    encrypted  = cipher.encrypt(compressed)
    return np.frombuffer(encrypted, dtype=np.uint8)


def unpack_parameters(arr: np.ndarray, cipher: ChaCha20Cipher) -> list:
    """Decrypt, decompress, and deserialize a packed uint8 array back to numpy arrays."""
    decrypted    = cipher.decrypt(arr.tobytes())   # raises ValueError if tampered
    decompressed = zlib.decompress(decrypted)
    return _deserialize_arrays(decompressed)


def is_packed(params_list: list) -> bool:
    """True when parameters are in packed (single uint8) form."""
    return len(params_list) == 1 and params_list[0].dtype == np.uint8