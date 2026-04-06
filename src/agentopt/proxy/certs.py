"""CA certificate generation for CONNECT-mode TLS termination.

On first use, generates a root CA key + self-signed certificate and stores
them in ``~/.agentopt/ca/``.  Per-hostname leaf certificates are generated
on demand and cached in memory.

The CA cert must be trusted by the agent process — the tracker sets
``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, and ``NODE_EXTRA_CA_CERTS``
environment variables to point at it.
"""

import logging
import ssl
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

_DEFAULT_CA_DIR = Path.home() / ".agentopt" / "ca"

# Certificate parameters.
_KEY_SIZE = 2048
_CA_VALIDITY_DAYS = 3650  # 10 years
_LEAF_VALIDITY_DAYS = 365  # 1 year


class CertificateAuthority:
    """Manages a local CA for MITM-style TLS termination of LLM API calls.

    Parameters
    ----------
    ca_dir : Path
        Directory for CA key and certificate PEM files.  Created on first use.
    """

    def __init__(self, ca_dir: Path = _DEFAULT_CA_DIR) -> None:
        self._ca_dir = ca_dir
        self._ca_key_path = ca_dir / "key.pem"
        self._ca_cert_path_val = ca_dir / "cert.pem"

        # Lazy-loaded.
        self._ca_key: rsa.RSAPrivateKey | None = None
        self._ca_cert: x509.Certificate | None = None

        # Per-hostname leaf cert cache: hostname -> (cert_pem, key_pem).
        self._leaf_cache: Dict[str, Tuple[bytes, bytes]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ca_cert_path(self) -> str:
        """Path to the CA certificate PEM file.  Generates if needed."""
        self._ensure_ca()
        return str(self._ca_cert_path_val)

    @property
    def ca_bundle_path(self) -> str:
        """Path to a merged CA bundle: system CAs + our CA.

        Subprocess agents need this so that:
        * LLM hosts (MITM'd by our proxy) verify against our CA.
        * Non-LLM hosts (passthrough) verify against system CAs.

        Without the merge, setting ``SSL_CERT_FILE`` to just our CA cert
        would break all non-LLM HTTPS connections.
        """
        self._ensure_ca()
        bundle_path = self._ca_dir / "bundle.pem"

        if bundle_path.exists():
            return str(bundle_path)

        with self._lock:
            if bundle_path.exists():
                return str(bundle_path)

            # Collect system CAs.
            system_certs = self._get_system_ca_certs()
            our_cert = self._ca_cert_path_val.read_bytes()

            bundle_path.write_bytes(system_certs + b"\n" + our_cert)
            logger.info("Created merged CA bundle at %s", bundle_path)
            return str(bundle_path)

    @staticmethod
    def _get_system_ca_certs() -> bytes:
        """Return the system CA certificate bundle as PEM bytes."""
        # Try certifi first (most reliable across platforms).
        try:
            import certifi

            return Path(certifi.where()).read_bytes()
        except ImportError:
            pass

        # Fall back to ssl's default CA paths.
        ctx = ssl.create_default_context()
        certs = ctx.get_ca_certs(binary_form=True)
        if certs:
            from cryptography.x509 import load_der_x509_certificate
            from cryptography.hazmat.primitives.serialization import Encoding

            pem_parts = []
            for der_cert in certs:
                cert = load_der_x509_certificate(der_cert)
                pem_parts.append(cert.public_bytes(Encoding.PEM))
            return b"\n".join(pem_parts)

        # Last resort: common system paths.
        for path in (
            "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
            "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/CentOS
            "/etc/ssl/cert.pem",  # macOS
        ):
            p = Path(path)
            if p.exists():
                return p.read_bytes()

        logger.warning(
            "Could not find system CA certs; bundle will only contain our CA"
        )
        return b""

    def get_server_context(self, hostname: str) -> ssl.SSLContext:
        """Return an ``ssl.SSLContext`` for serving TLS as *hostname*.

        The returned context uses a leaf certificate signed by our CA
        with a SAN matching *hostname*.
        """
        self._ensure_ca()
        cert_pem, key_pem = self._get_leaf_cert(hostname)

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(
            certfile=_write_temp_pem(cert_pem, self._ca_dir, f"{hostname}.cert.pem"),
            keyfile=_write_temp_pem(key_pem, self._ca_dir, f"{hostname}.key.pem"),
        )
        return ctx

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_ca(self) -> None:
        """Load or generate the root CA key and certificate."""
        if self._ca_key is not None:
            return

        with self._lock:
            if self._ca_key is not None:
                return  # Double-check after acquiring lock.

            self._ca_dir.mkdir(parents=True, exist_ok=True)

            if self._ca_key_path.exists() and self._ca_cert_path_val.exists():
                self._ca_key = serialization.load_pem_private_key(
                    self._ca_key_path.read_bytes(), password=None,
                )
                self._ca_cert = x509.load_pem_x509_certificate(
                    self._ca_cert_path_val.read_bytes(),
                )
                logger.debug("Loaded existing CA from %s", self._ca_dir)
            else:
                self._ca_key, self._ca_cert = _generate_ca()
                self._ca_key_path.write_bytes(
                    self._ca_key.private_bytes(
                        serialization.Encoding.PEM,
                        serialization.PrivateFormat.TraditionalOpenSSL,
                        serialization.NoEncryption(),
                    )
                )
                self._ca_cert_path_val.write_bytes(
                    self._ca_cert.public_bytes(serialization.Encoding.PEM)
                )
                logger.info("Generated new CA certificate in %s", self._ca_dir)

    def _get_leaf_cert(self, hostname: str) -> Tuple[bytes, bytes]:
        """Return ``(cert_pem, key_pem)`` for *hostname*, generating if needed."""
        cached = self._leaf_cache.get(hostname)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._leaf_cache.get(hostname)
            if cached is not None:
                return cached

            assert self._ca_key is not None and self._ca_cert is not None
            cert_pem, key_pem = _generate_leaf(hostname, self._ca_key, self._ca_cert,)
            self._leaf_cache[hostname] = (cert_pem, key_pem)
            return cert_pem, key_pem


# ------------------------------------------------------------------
# Certificate helpers
# ------------------------------------------------------------------


def _generate_ca() -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generate a self-signed root CA key and certificate."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=_KEY_SIZE)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AgentOpt"),
            x509.NameAttribute(NameOID.COMMON_NAME, "AgentOpt Local CA"),
        ]
    )

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _generate_leaf(
    hostname: str, ca_key: rsa.RSAPrivateKey, ca_cert: x509.Certificate,
) -> Tuple[bytes, bytes]:
    """Generate a leaf certificate signed by the CA for *hostname*."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=_KEY_SIZE)

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname),]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_LEAF_VALIDITY_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key(),),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def _write_temp_pem(data: bytes, directory: Path, filename: str) -> str:
    """Write PEM data to a file in *directory*, returning the path."""
    path = directory / filename
    path.write_bytes(data)
    return str(path)
