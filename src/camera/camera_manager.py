import cv2
import socket
from urllib.parse import urlparse


class CamaraManager:
	"""Gestor simple de cámara.

	- Abre la webcam con id 0 por defecto.
	- get_frame devuelve el cuadro espejado si la lectura fue exitosa.
	- release libera el acceso a la cámara.
	- También puede abrir una cámara IP si se provee una URL.
	"""

	def __init__(self, webcam_id: int = 0, ip_url: str | None = None):
		self.webcam_id = webcam_id
		self.ip_url = (ip_url or "").strip().strip('"').strip("'") or None
		self.cap: cv2.VideoCapture | None = None

	def _check_ip_endpoint(self, source: str, timeout: float = 2.0) -> None:
		parsed = urlparse(source)
		host = parsed.hostname
		if not host:
			raise RuntimeError(f"URL de cámara IP inválida: {source}")
		port = parsed.port
		if port is None:
			port = 443 if parsed.scheme == "https" else 80
		try:
			with socket.create_connection((host, port), timeout=timeout):
				return
		except OSError as exc:
			raise RuntimeError(
				f"No hay conexión TCP a {host}:{port}. Verifica red/WiFi y app de cámara IP."
			) from exc

	def open(self) -> None:
		import time
		
		source = self.ip_url if self.ip_url else self.webcam_id

		# Para cámaras IP, dar tiempo para establecer conexión
		if self.ip_url:
			print(f"Conectando a cámara IP: {source}")
			self._check_ip_endpoint(source)
			# Reintentar verificación con backoff para cámaras IP
			max_attempts = 5
			for attempt in range(max_attempts):
				try:
					if self.cap is not None:
						self.cap.release()
				except Exception:
					pass

				self.cap = cv2.VideoCapture(source)

				if self.cap.isOpened():
					ok, _ = self.cap.read()
					if ok:
						print("✓ Conexión establecida")
						return  # conexión exitosa
				if attempt < max_attempts - 1:
					wait_time = 0.5 * (attempt + 1)
					print(f"  Reintento {attempt + 1}/{max_attempts - 1} en {wait_time:.1f}s...")
					time.sleep(wait_time)
			
			# Si llegamos aquí, falló
			if self.cap:
				self.cap.release()
				self.cap = None
			raise RuntimeError(f"No se pudo conectar a la cámara IP: {source}\n"
			                   f"Verifica que:\n"
			                   f"  1. El celular esté en la misma red WiFi\n"
			                   f"  2. La app de cámara IP esté ejecutándose\n"
			                   f"  3. La URL sea correcta (prueba abrirla en el navegador)")
		else:
			if self.cap is None:
				self.cap = cv2.VideoCapture(source)
			# Para cámaras locales, verificar inmediatamente
			if not self.cap or not self.cap.isOpened():
				raise RuntimeError(f"No se pudo abrir la cámara local: {source}")

	def get_frame(self):
		"""Lee un frame en tiempo real.

		Retorna (True, frame_espejado) si tuvo éxito, o (False, None) si falló.
		"""
		if self.cap is None or not self.cap.isOpened():
			self.open()
		success, frame = self.cap.read()
		if not success:
			return False, None
		# Espejar horizontalmente
		mirrored = cv2.flip(frame, 1)
		return True, mirrored

	def release(self) -> None:
		if self.cap is not None:
			self.cap.release()
			self.cap = None
