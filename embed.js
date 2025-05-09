/* --------------------------------------------------------------------------
 *  embed.js  —  Lee un PDF, lo trocea, genera embeddings con OpenAI
 *               y los guarda en el vector‑store local (JSON).
 * -------------------------------------------------------------------------- */

import fs from "fs/promises";
import path from "path";
import crypto from "crypto";
import pdf from "pdf-parse-debugging-disabled";   // ← fork sin modo debug
import { OpenAI } from "openai";
import { upsert } from "./utils/vectorStore.js";
import { fileURLToPath } from 'url';

/* ---------- parámetros ---------- */
const EMBED_MODEL  = "text-embedding-3-small"; // modelo de embeddings
const CHUNK_WORDS  = 500;                      // 500 palabras aprox. ~ 750‑800 tokens
const OVERLAP      = 50;                       // superposición para contexto suave

// Configuración de directorios
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_MANUALS_DIR = path.join(__dirname, 'public/manuals');

/* ---------- helper: corta el texto en trozos ---------- */
function chunkText(txt) {
  const words  = txt.split(/\s+/);
  const blocks = [];

  for (let i = 0; i < words.length; i += CHUNK_WORDS - OVERLAP) {
    const slice = words.slice(i, i + CHUNK_WORDS).join(" ");
    blocks.push(slice);
  }

  return blocks;
}

/* ---------- función principal ---------- */
export async function ingestPDF(pathPdf, originalFilename) {
  /* 1) leer y extraer texto */
  const buffer = await fs.readFile(pathPdf);
  const pdfData = await pdf(buffer);
  const text = pdfData.text;
  const info = pdfData.info || {};

  /* 2) trocear en bloques */
  const pieces = chunkText(text);

  /* 3) generar embeddings */
  const openai = new OpenAI();
  const vectors = [];

  for (const [idx, piece] of pieces.entries()) {
    const { data: [embed] } = await openai.embeddings.create({
      model: EMBED_MODEL,
      input: piece
    });

    vectors.push({
      id:     `chunk-${idx}`,
      text:   piece,
      vector: embed        // array de floats
    });
  }

  /* 4) Generar un ID único para el manual */
  const filename = originalFilename || path.basename(pathPdf);
  const manualId = crypto.createHash('md5').update(filename + Date.now()).digest('hex');
  
  /* 5) Preparar información del manual */
  const manualInfo = {
    filename: filename,
    title: info.Title || filename,
    author: info.Author || "Desconocido",
    pages: pdfData.numpages || 0,
    createdAt: new Date().toISOString(),
    path: `/manuals/${manualId}.pdf`  // Ruta donde se guardará el PDF
  };

  /* 6) Asegurar que exista el directorio de manuales público */
  await fs.mkdir(PUBLIC_MANUALS_DIR, { recursive: true });
  
  /* 7) Copiar el PDF a la carpeta pública de manuales */
  await fs.copyFile(pathPdf, path.join(PUBLIC_MANUALS_DIR, `${manualId}.pdf`));
  
  /* 8) guardar todo en el vector store */
  await upsert(vectors, manualId, manualInfo);

  /* 9) devolver información del manual indexado */
  return {
    id: manualId,
    chunks: pieces.length,
    info: manualInfo
  };
}