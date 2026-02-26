import cv2


class CamaraManager:
	"""Gestor simple de cámara.

	- Abre la webcam con id 0 por defecto.
	- get_frame devuelve el cuadro espejado si la lectura fue exitosa.
	- release libera el acceso a la cámara.
	- También puede abrir una cámara IP si se provee una URL.
	"""

	def __init__(self, webcam_id: int = 0, ip_url: str | None = None):
		self.webcam_id = webcam_id
		self.ip_url = ip_url
		self.cap: cv2.VideoCapture | None = None

	def open(self) -> None:
		if self.cap is None:
			source = self.ip_url if self.ip_url else self.webcam_id
			self.cap = cv2.VideoCapture(source)
		if not self.cap or not self.cap.isOpened():
			raise RuntimeError("No se pudo abrir la cámara")

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


# Mantener compatibilidad con código existente
class CameraManager(CamaraManager):
	pass