// utils/vectorStore.js
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, "../data");
const MANUALS_FILE = path.join(DATA_DIR, "manuals.json");

let cache = null;

/* ---------- helpers ---------- */ 

// carga de disco a RAM (lazy)
async function load() {
  if (cache) return cache;          // ya en memoria
  try {
    const json = await fs.readFile(MANUALS_FILE, "utf8");
    cache = JSON.parse(json);
  } catch {
    cache = {
      manuals: {}                   // estructura: { manualId: { embeddings, info } }
    };                     
  }
  return cache;
}

// guarda la versión en RAM en disco
async function save() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.writeFile(MANUALS_FILE, JSON.stringify(cache), "utf8");
}

/* ---------- API ---------- */

export async function upsert(embeddings, manualId, manualInfo) {
  // embeddings = [{ id, text, vector }]
  await load();
  
  if (!cache.manuals) {
    cache.manuals = {};
  }
  
  // Guardamos los embeddings y la información del manual
  cache.manuals[manualId] = {
    embeddings: embeddings,
    info: manualInfo
  };
  
  await save();
  return true;
}

export async function similaritySearch(queryVec, topK = 4) {
  await load();
  if (!cache.manuals || Object.keys(cache.manuals).length === 0) return [];

  const cosine = (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na  += a[i] * a[i];
      nb  += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  };

  // Lista para almacenar todos los resultados
  let allResults = [];

  // Buscar en todos los manuales
  for (const manualId in cache.manuals) {
    const manualData = cache.manuals[manualId];
    const embeddings = manualData.embeddings;
    const manualInfo = manualData.info;

    for (const obj of embeddings) {
      const score = cosine(queryVec, obj.vector);
      
      // CAMBIO CLAVE: Reducir el umbral de relevancia a 0.4 (era 0.65)
      // Esto permitirá que más resultados sean considerados relevantes
      if (score > 0.4) {
        allResults.push({
          text: obj.text,
          score: score,
          manualId: manualId,
          manualInfo: manualInfo,
          chunkId: obj.id || `chunk-${embeddings.indexOf(obj)}`
        });
      }
    }
  }

  // Si no hay resultados que superen el umbral reducido, tomamos los mejores sin filtrar
  if (allResults.length === 0) {
    console.log("No se encontraron resultados por encima del umbral, tomando los mejores disponibles...");
    
    // Buscar en todos los manuales sin umbral
    for (const manualId in cache.manuals) {
      const manualData = cache.manuals[manualId];
      const embeddings = manualData.embeddings;
      const manualInfo = manualData.info;

      for (const obj of embeddings) {
        const score = cosine(queryVec, obj.vector);
        allResults.push({
          text: obj.text,
          score: score,
          manualId: manualId,
          manualInfo: manualInfo,
          chunkId: obj.id || `chunk-${embeddings.indexOf(obj)}`
        });
      }
    }
  }

  // Ordenar todos los resultados por puntuación y tomar los mejores
  return allResults
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// Función para obtener la lista de manuales disponibles
export async function getAvailableManuals() {
  await load();
  
  if (!cache.manuals) {
    return [];
  }
  
  const manuals = [];
  for (const manualId in cache.manuals) {
    manuals.push({
      id: manualId,
      info: cache.manuals[manualId].info,
      chunks: cache.manuals[manualId].embeddings.length
    });
  }
  
  return manuals;
}

// Verificar si hay algún manual cargado
export async function verifyVectorStore() {
  try {
    await load();
    
    if (!cache.manuals) {
      return [];
    }
    
    let allEmbeddings = [];
    for (const manualId in cache.manuals) {
      allEmbeddings = allEmbeddings.concat(cache.manuals[manualId].embeddings);
    }
    
    return allEmbeddings;
  } catch (err) {
    console.error("Error al verificar vector store:", err);
    return [];
  }
}

// Nueva función para obtener metadatos de un chunk específico
export async function getChunkMetadata(manualId, chunkId) {
  await load();
  
  if (!cache.manuals || !cache.manuals[manualId]) {
    return null;
  }
  
  const chunk = cache.manuals[manualId].embeddings.find(emb => emb.id === chunkId);
  if (!chunk) {
    return null;
  }
  
  // Devolvemos metadatos del chunk sin incluir el vector completo (que es muy grande)
  return {
    id: chunk.id,
    text: chunk.text.substring(0, 100) + "...", // Solo un preview del texto
    manualId: manualId,
    manualInfo: cache.manuals[manualId].info
  };
}