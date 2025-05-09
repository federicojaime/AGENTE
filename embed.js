/* --------------------------------------------------------------------------
 *  embed.js  —  Lee un PDF, lo trocea, genera embeddings con OpenAI
 *               y los guarda en el vector‑store local (JSON).
 * -------------------------------------------------------------------------- */

import fs from "fs/promises";
import pdf from "pdf-parse-debugging-disabled";   // ← fork sin modo debug
import { OpenAI } from "openai";
import { upsert } from "./utils/vectorStore.js";

/* ---------- parámetros ---------- */
const EMBED_MODEL  = "text-embedding-3-small"; // modelo de embeddings
const CHUNK_WORDS  = 500;                      // 500 palabras aprox. ~ 750‑800 tokens
const OVERLAP      = 50;                       // superposición para contexto suave

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
export async function ingestPDF(pathPdf) {
  /* 1) leer y extraer texto */
  const buffer   = await fs.readFile(pathPdf);
  const { text } = await pdf(buffer);

  /* 2) trocear en bloques */
  const pieces = chunkText(text);

  /* 3) generar embeddings */
  const openai  = new OpenAI();
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

  /* 4) guardar todo el manual (sobrescribe) */
  await upsert(vectors);

  /* 5) devolver cuántos chunks se indexaron */
  return pieces.length;
}
