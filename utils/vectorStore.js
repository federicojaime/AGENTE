// utils/vectorStore.js
import fs from "fs/promises";
import path from "path";

const DATA_DIR = path.resolve("data");
const FILE = path.join(DATA_DIR, "manual.json");

let cache = null;

/* ---------- helpers ---------- */

// carga de disco a RAM (lazy)
async function load() {
  if (cache) return cache;          // ya en memoria
  try {
    const json = await fs.readFile(FILE, "utf8");
    cache = JSON.parse(json);
  } catch {
    cache = [];                     // no existe aún
  }
  return cache;
}

// guarda la versión en RAM en disco
async function save() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.writeFile(FILE, JSON.stringify(cache), "utf8");
}

/* ---------- API ---------- */

export async function upsert(embeddings) {
  // embeddings = [{ id, text, vector }]
  await load();
  cache = embeddings;    // sobrescribimos (un solo manual)
  await save();
}

export async function similaritySearch(queryVec, topK = 4) {
  await load();
  if (!cache.length) return [];

  const cosine = (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na  += a[i] * a[i];
      nb  += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  };

  const scored = cache.map(obj => ({
    text: obj.text,
    score: cosine(queryVec, obj.vector)
  }));

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(s => s.text);
}

// Nueva función para verificar si hay un vector store
export async function verifyVectorStore() {
  try {
    await load();
    return cache;
  } catch (err) {
    console.error("Error al verificar vector store:", err);
    return [];
  }
}

// Nueva función para limpiar el store (útil para pruebas)
export async function clearVectorStore() {
  try {
    cache = [];
    await save();
    return true;
  } catch (err) {
    console.error("Error al limpiar vector store:", err);
    return false;
  }
}

// Función para obtener estadísticas del vector store
export async function getVectorStoreStats() {
  try {
    await load();
    
    // Información básica
    const stats = {
      totalChunks: cache.length,
      averageChunkLength: 0,
      totalTokensEstimated: 0,
      firstChunkPreview: cache.length > 0 ? cache[0].text.substring(0, 100) + "..." : "",
      lastUpdated: null
    };
    
    // Calculamos longitud promedio
    if (cache.length > 0) {
      let totalLength = 0;
      for (const chunk of cache) {
        totalLength += chunk.text.length;
      }
      stats.averageChunkLength = Math.round(totalLength / cache.length);
      stats.totalTokensEstimated = Math.round(totalLength / 4); // Estimación aproximada de tokens
    }
    
    // Intentamos obtener la fecha de última modificación
    try {
      const fileStats = await fs.stat(FILE);
      stats.lastUpdated = fileStats.mtime.toISOString();
    } catch (err) {
      stats.lastUpdated = null;
    }
    
    return stats;
  } catch (err) {
    console.error("Error al obtener estadísticas:", err);
    return null;
  }
}