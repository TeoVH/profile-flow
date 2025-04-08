import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function Vacantes() {
  const [vacantes, setVacantes] = useState([]);
  const [usuarioActual, setUsuarioActual] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDatos = async () => {
      try {
        const token = localStorage.getItem("token");

        const [vacantesRes, userRes] = await Promise.all([
          axios.get("http://localhost:8000/api/jobs/", {
            headers: { Authorization: `Token ${token}` },
          }),
          axios.get("http://localhost:8000/api/users/me/", {
            headers: { Authorization: `Token ${token}` },
          }),
        ]);

        setVacantes(vacantesRes.data);
        setUsuarioActual(userRes.data);
      } catch (error) {
        console.error("Error al obtener datos:", error);
      }
    };

    fetchDatos();
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1><strong>Vacantes Disponibles</strong></h1>

      {usuarioActual?.is_recruiter && (
        <button
          onClick={() => navigate("/crear-vacante")}
          style={{
            marginBottom: "1.5rem",
            padding: "0.5rem 1rem",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Crear nueva vacante
        </button>
      )}

      <ul style={{ listStyle: "none", padding: 0 }}>
        {vacantes.map((vacante) => (
          <li
            key={vacante.id}
            style={{
              border: "1px solid #ddd",
              borderRadius: "6px",
              padding: "1rem",
              marginBottom: "1rem",
              backgroundColor: "#f9f9f9"
            }}
          >
            <h3>{vacante.title}</h3>
            <p>{vacante.description}</p>
            <p><em>Ubicación:</em> {vacante.location}</p>
            <p><em>Empresa:</em> {vacante.company}</p>
            <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
              <button
                onClick={() => navigate(`/vacantes/${vacante.id}`)}
                style={{
                  padding: "0.4rem 0.8rem",
                  border: "1px solid #007bff",
                  backgroundColor: "#007bff",
                  color: "white",
                  borderRadius: "4px",
                  cursor: "pointer"
                }}
              >
                Ver detalle
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Vacantes;
