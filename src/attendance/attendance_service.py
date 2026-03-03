from __future__ import annotations
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict


class AttendanceService:
    """
    Servicio para manejar el registro de asistencia en un archivo CSV. Permite marcar la asistencia de una persona con un estado (e.g. "present") y evita registros duplicados dentro de un período de tiempo definido (dedupe_seconds).
     - csv_path: ruta al archivo CSV donde se guardarán los registros de asistencia.
     - dedupe_seconds: número de segundos para considerar un registro como duplicado (default: 300 segundos = 5 minutos).
     - mark_attendance(name, status): marca la asistencia de una persona. Devuelve True si se escribió un nuevo registro, o False si se omitió por ser un duplicado reciente.
     - read_all(): lee todos los registros de asistencia y devuelve un DataFrame de pandas.
    """


    def __init__(self, csv_path: str = "data/deepface/attendance/attendance.csv", dedupe_seconds: int = 300):
        self.csv_path = csv_path
        self.dedupe_seconds = dedupe_seconds
        self.last_seen: Dict[str, datetime] = {}
        # ensure folder exists
        folder = os.path.dirname(csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    def _recently_marked(self, name: str) -> bool:
        now = datetime.now()
        if name in self.last_seen:
            if (now - self.last_seen[name]).total_seconds() < self.dedupe_seconds:
                return True
        return False

    def mark_attendance(self, name: str, status: str = "present") -> bool:
        """Marca la asistencia de una persona. Devuelve True si se escribió un nuevo registro, o False si se omitió por ser un duplicado reciente."""
        if self._recently_marked(name):
            return False
        now = datetime.now()
        row = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "name": name,
            "status": status,
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=header, index=False)
        self.last_seen[name] = now
        return True

    def read_all(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=["date", "time", "name", "status"])
        return pd.read_csv(self.csv_path)


__all__ = ["AttendanceService"]
