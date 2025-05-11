/* --------------------------------------------------------------------------
 *  embed.js  —  Lee un PDF, lo trocea, genera embeddings con OpenAI
 *               y los guarda en el vector‑store local (JSON).
 * -------------------------------------------------------------------------- */

import fs from "fs/promises";
import path from "path";
import crypto from "crypto";
import axios from "axios";
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
function chunkText(txt, metadata = {}) {
  const words = txt.split(/\s+/);
  const blocks = [];
  
  // Extraer información de estructura del PDF si está disponible
  const hasToc = metadata.outline && metadata.outline.length > 0;
  const tocEntries = hasToc ? metadata.outline : [];
  
  // Crear un mapeo aproximado de palabras a secciones basado en el TOC si existe
  let sections = [];
  if (hasToc) {
    sections = tocEntries.map(entry => ({
      title: entry.title,
      approxWordPos: Math.floor(entry.pageNum * words.length / metadata.pages)
    }));
  }

  for (let i = 0; i < words.length; i += CHUNK_WORDS - OVERLAP) {
    const slice = words.slice(i, i + CHUNK_WORDS).join(" ");
    
    // Determinar si este bloque pertenece a alguna sección específica
    let sectionInfo = "";
    if (sections.length > 0) {
      // Encontrar la sección más cercana anterior a esta posición de palabra
      const currentSection = sections
        .filter(s => s.approxWordPos <= i)
        .sort((a, b) => b.approxWordPos - a.approxWordPos)[0];
      
      if (currentSection) {
        sectionInfo = `[Sección: ${currentSection.title}]`;
      }
    }
    
    // Añadir información de sección al inicio del fragmento si existe
    const blockText = sectionInfo ? `${sectionInfo}\n${slice}` : slice;
    
    blocks.push(blockText);
  }

  return blocks;
}

/* ---------- función principal para PDFs locales ---------- */
export async function ingestPDF(pathPdf, originalFilename) {
  /* 1) leer y extraer texto */
  const buffer = await fs.readFile(pathPdf);
  const pdfData = await pdf(buffer);
  const text = pdfData.text;
  const info = pdfData.info || {};
  
  // Extraer metadatos avanzados
  const metadata = {
    pages: pdfData.numpages || 0,
    outline: pdfData.outline || [],
    info: info
  };

  /* 2) trocear en bloques */
  const pieces = chunkText(text, metadata);

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
    path: `/manuals/${manualId}.pdf`,  // Ruta donde se guardará el PDF
    isRemote: false,
    outline: pdfData.outline || [] // Guardar estructura de secciones si existe
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

/* ---------- función para procesar PDF remoto ---------- */
export async function ingestRemotePDF(url, title, metadata = {}) {
  console.log(`Descargando PDF desde: ${url}`);
  
  /* 1) Descargar el PDF */
  const response = await axios.get(url, { 
    responseType: 'arraybuffer',
    headers: {
      'Accept': 'application/pdf'
    },
    maxContentLength: 50 * 1024 * 1024 // 50 MB max
  });
  const buffer = Buffer.from(response.data);
  
  /* 2) Extraer texto */
  const pdfData = await pdf(buffer);
  const text = pdfData.text;
  const info = pdfData.info || {};
  
  // Extraer metadatos avanzados
  const pdfMetadata = {
    pages: pdfData.numpages || 0,
    outline: pdfData.outline || [],
    info: info
  };
  
  /* 3) Trocear en bloques */
  const pieces = chunkText(text, pdfMetadata);
  
  /* 4) Generar embeddings */
  const openai = new OpenAI();
  const vectors = [];
  
  for (const [idx, piece] of pieces.entries()) {
    const { data: [embed] } = await openai.embeddings.create({
      model: EMBED_MODEL,
      input: piece
    });
    
    vectors.push({
      id: `chunk-${idx}`,
      text: piece,
      vector: embed
    });
  }
  
  /* 5) Generar un ID único para el manual */
  const manualId = crypto.createHash('md5').update(url + Date.now()).digest('hex');
  
  /* 6) Preparar información del manual */
  const filename = url.split("/").pop().split("?")[0] || "documento.pdf";
  const manualInfo = {
    url: url,
    filename: filename,
    title: title || metadata.title || info.Title || filename,
    author: metadata.author || info.Author || "Desconocido",
    pages: pdfData.numpages || 0,
    createdAt: new Date().toISOString(),
    path: url, // La URL original para acceder directamente
    isRemote: true,
    outline: pdfData.outline || [] // Guardar estructura de secciones si existe
  };
  
  /* 7) Guardar en el vector store */
  await upsert(vectors, manualId, manualInfo);
  
  /* 8) Devolver información */
  return {
    id: manualId,
    chunks: pieces.length,
    info: manualInfo
  };
}